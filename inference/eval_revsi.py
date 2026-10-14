#!/usr/bin/env python3
"""ReVSI-Bench, 32-frame protocol.

Place the release under ``../data/ReVSI`` (see the README). Questions come from
``32_frame/test.jsonl`` and videos from ``32_frame/<scene_id>.mp4``.

Prompts and scores follow the training eval: multiple-choice is an exact letter
match, numeric answers use mean relative accuracy over thresholds 0.5–0.95, and
the headline averages question types after merging the direction, relative-distance,
counting, and room-size subtypes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from eval_utils import (
    fuzzy_matching,
    gather_results,
    mean_relative_accuracy,
    setup_distributed,
    to_float,
)
from dkvlm import load_dkvlm

MCQ_QUESTION_TYPES = {
    "object_rel_direction_forward_easy",
    "object_rel_direction_backward_easy",
    "object_rel_direction_forward_hard",
    "object_rel_direction_backward_hard",
    "object_rel_distance_closest",
    "object_rel_distance_farthest",
    "route_planning",
}
NQ_QUESTION_TYPES = {
    "object_counting_single",
    "object_counting_multiple",
    "object_abs_distance",
    "object_size_estimation",
    "room_size_estimation_single",
    "room_size_estimation_multiple",
}
COMPOSITE_METRICS = {
    "object_rel_direction_acc": [
        "object_rel_direction_forward_easy",
        "object_rel_direction_backward_easy",
        "object_rel_direction_forward_hard",
        "object_rel_direction_backward_hard",
    ],
    "object_rel_distance_acc": [
        "object_rel_distance_closest",
        "object_rel_distance_farthest",
    ],
    "object_counting_acc": [
        "object_counting_single",
        "object_counting_multiple",
    ],
    "room_size_estimation_acc": [
        "room_size_estimation_single",
        "room_size_estimation_multiple",
    ],
}
DISPLAY_KEYS = {
    "object_abs_distance_acc": "abs_distance",
    "object_size_estimation_acc": "size",
    "object_rel_distance_acc": "rel_distance",
    "room_size_estimation_acc": "room",
    "object_rel_direction_acc": "direction",
    "route_planning_acc": "route",
    "object_counting_acc": "counting",
}
MRA_METRIC = "MRA:.5:.95:.05"


def as_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(x) for x in value]


def revsi_prompt(item: Dict[str, Any]) -> str:
    question = str(item.get("question") or "")
    pre = "These are frames of a video."
    qtype = str(item.get("question_type") or "")
    if qtype in NQ_QUESTION_TYPES:
        post = "Answer the question using a single integer or decimal number."
        return "\n".join([pre, question, post]).strip()
    if qtype in MCQ_QUESTION_TYPES:
        options = "Options:\n" + "\n".join(as_str_list(item.get("options")))
        post = "Answer with the option's letter from the given choices directly."
        return "\n".join([pre, question, options, post]).strip()
    raise ValueError(f"Unknown ReVSI question type: {qtype}")


def score_sample(item: Dict[str, Any], output_text: str) -> tuple[str, float]:
    qtype = item.get("question_type")
    gt = item.get("ground_truth", "")
    pred = fuzzy_matching(output_text)
    if qtype in MCQ_QUESTION_TYPES:
        return "accuracy", 1.0 if str(pred).lower() == str(gt).lower() else 0.0
    if qtype in NQ_QUESTION_TYPES:
        pred_f = to_float(pred)
        gt_f = to_float(fuzzy_matching(gt))
        if pred_f is None or gt_f is None:
            return MRA_METRIC, 0.0
        return MRA_METRIC, mean_relative_accuracy(pred_f, gt_f)
    raise ValueError(f"Unknown ReVSI question type: {qtype}")


def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    acc_by_q: Dict[str, List[float]] = defaultdict(list)
    for row in results:
        qtype = str(row.get("question_type") or "")
        val = row.get("metric_value")
        if qtype and val is not None:
            acc_by_q[qtype].append(float(val))
    per_type = {f"{qtype}_acc": float(np.mean(vals)) if vals else 0.0 for qtype, vals in acc_by_q.items()}
    collapsed = dict(per_type)
    for metric_name, qtypes in COMPOSITE_METRICS.items():
        keys = [f"{q}_acc" for q in qtypes if f"{q}_acc" in collapsed]
        if not keys:
            continue
        collapsed[metric_name] = float(np.mean([collapsed.pop(k) for k in keys]))
    overall = float(np.mean(list(collapsed.values()))) if collapsed else 0.0
    raw_type_mean = float(np.mean(list(per_type.values()))) if per_type else 0.0
    metrics: Dict[str, Any] = {
        "overall": overall,
        "overall_score_percent": overall * 100.0,
        "raw_type_mean": raw_type_mean,
        "raw_type_mean_percent": raw_type_mean * 100.0,
    }
    metrics.update(collapsed)
    for src, dst in DISPLAY_KEYS.items():
        if src in collapsed:
            metrics[dst] = float(collapsed[src]) * 100.0
    metrics["per_question_type"] = {k: float(v) for k, v in per_type.items()}
    return metrics


def video_path_for_item(item: Dict[str, Any], revsi_dir: Path) -> Path:
    scene = str(item.get("scene_id") or item.get("scene_name") or "").strip()
    return revsi_dir / "32_frame" / f"{scene}.mp4"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--base_model_path", required=True, help="stock Qwen-VL directory")
    ap.add_argument("--geometry_ckpt", default="")
    ap.add_argument("--revsi_dir", required=True, help="directory that contains 32_frame/test.jsonl and 32_frame/<scene_id>.mp4")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--nframes", type=int, default=32)
    ap.add_argument("--max_samples", type=int, default=-1, help="-1 evaluates every row in test.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--attn_implementation", default="flash_attention_2", choices=["eager", "sdpa", "flash_attention_2"])
    ap.add_argument("--geo_attention_backend", default="sdpa", choices=["eager", "sdpa", "chunked"])
    ap.add_argument(
        "--scene_cache_size",
        type=int,
        default=32,
        help="Per-video cache of decoded frames + VGGT geometry (0 disables)",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    device, rank, world_size = setup_distributed()
    out_dir = Path(args.output_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    revsi_dir = Path(args.revsi_dir)
    jsonl = revsi_dir / "32_frame" / "test.jsonl"
    items = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    total = len(items)
    if args.max_samples > 0:
        items = items[: args.max_samples]

    model = load_dkvlm(
        args.checkpoint_dir,
        base_model_path=args.base_model_path,
        geometry_ckpt=args.geometry_ckpt,
        device=device,
        attn_implementation=args.attn_implementation,
        geo_attention_backend=args.geo_attention_backend,
        nframes=args.nframes,
        use_geometry=True,
        verbose=(rank == 0),
    )
    model.max_new_tokens = int(args.max_new_tokens)
    ip = model.processor.image_processor
    patch = int(getattr(ip, "patch_size", 0) or 0)
    merge = int(getattr(ip, "merge_size", 0) or 0)
    cell = patch * merge
    protocol = "revsi_32_frame_qwen25_7b_cell28" if cell == 28 else f"revsi_32_frame_cell{cell}"

    shard = items[rank::world_size]
    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    scene_cache: Dict[str, Any] = {}
    scene_cache_order: List[str] = []
    cache_hits = 0
    cache_misses = 0
    started = time.perf_counter()

    def cached_scene(video: Path, scene: str):
        nonlocal cache_hits, cache_misses
        key = str(video)
        if args.scene_cache_size > 0 and key in scene_cache:
            cache_hits += 1
            return scene_cache[key]
        sampled = model.resolve_frames(video)
        geometry = model.compute_geometry(sampled, sample_id=scene)
        cache_misses += 1
        if args.scene_cache_size > 0:
            scene_cache[key] = (sampled, geometry)
            scene_cache_order.append(key)
            while len(scene_cache_order) > args.scene_cache_size:
                scene_cache.pop(scene_cache_order.pop(0), None)
        return sampled, geometry

    for i, item in enumerate(shard, start=1):
        scene = str(item.get("scene_id") or item.get("scene_name") or "")
        try:
            video = video_path_for_item(item, revsi_dir)
            if not video.is_file():
                raise FileNotFoundError(f"video not found: {video}")
            sampled, geometry = cached_scene(video, scene)
            output_text = model.generate(
                model.build_inputs(sampled, revsi_prompt(item), geometry=geometry, sample_id=scene)
            )
            metric_name, metric_value = score_sample(item, output_text)
            results.append(
                {
                    "id": item.get("id"),
                    "question_type": item.get("question_type"),
                    "ground_truth": item.get("ground_truth"),
                    "prediction": output_text,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "scene_id": scene,
                }
            )
        except Exception as exc:
            failures.append({"id": item.get("id"), "scene_id": scene, "error": repr(exc)})

        if rank == 0 and (i % 25 == 0 or i == len(shard)):
            elapsed = time.perf_counter() - started
            rate = i / max(elapsed, 1e-6)
            eta = (len(shard) - i) / max(rate, 1e-6)
            print(f"[revsi] rank0 {i}/{len(shard)}  {rate:.2f} samples/s  eta {eta / 60:.1f} min", flush=True)

    gathered = gather_results({"rank": rank, "results": results, "failures": failures}, world_size)
    if rank == 0:
        all_results: List[Dict[str, Any]] = []
        all_failures: List[Dict[str, Any]] = []
        for part in sorted(gathered, key=lambda p: int(p.get("rank", 0))):
            all_results.extend(part.get("results") or [])
            all_failures.extend(part.get("failures") or [])
        metrics = aggregate_metrics(all_results)
        evaluation_complete = (
            bool(items) and not all_failures and len(all_results) == len(items)
        )
        full_dataset_evaluated = evaluation_complete and len(items) == total
        summary = {
            "metrics": metrics,
            "evaluation_status": "complete" if evaluation_complete else "incomplete",
            "evaluation_complete": evaluation_complete,
            "full_dataset_evaluated": full_dataset_evaluated,
            "score_scope": "full_dataset" if full_dataset_evaluated else (
                "requested_subset" if evaluation_complete else "successful_samples_only"
            ),
            "num_scored": len(all_results),
            "num_failed": len(all_failures),
            "num_requested": len(items),
            "dataset_total": total,
            "world_size": world_size,
            "config": {
                "protocol_name": protocol,
                "nframes": args.nframes,
                "max_new_tokens": args.max_new_tokens,
                "max_pixels": 786432,
                "patch": patch,
                "merge": merge,
                "cell": cell,
                "attn_implementation": args.attn_implementation,
                "geo_attention_backend": args.geo_attention_backend,
                "scene_cache_size": args.scene_cache_size,
                "scene_cache_hits": cache_hits,
                "scene_cache_misses": cache_misses,
            },
        }
        (out_dir / "predictions.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in all_results) + ("\n" if all_results else ""),
            encoding="utf-8",
        )
        (out_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if all_failures:
            (out_dir / "failures.jsonl").write_text(
                "\n".join(json.dumps(f, ensure_ascii=False) for f in all_failures) + "\n",
                encoding="utf-8",
            )
        print(json.dumps({"protocol_name": protocol, "overall_score_percent": metrics.get("overall_score_percent")}, ensure_ascii=False))
        if not evaluation_complete:
            print(
                "[revsi] WARNING: evaluation incomplete; scores cover successful "
                "samples only, not a full benchmark result."
            )
        elif not full_dataset_evaluated:
            print("[revsi] NOTE: this is a subset evaluation, not a full benchmark result.")
        print(f"[revsi] overall = {metrics['overall_score_percent']:.2f}%")
        print(f"[revsi] scored {len(all_results)} samples, {len(all_failures)} failed")

    if world_size > 1:
        import torch.distributed as dist

        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
