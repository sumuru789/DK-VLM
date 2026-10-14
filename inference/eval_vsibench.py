#!/usr/bin/env python3
"""Reproduce the VSI-Bench numbers reported for DK-VLM.

Single GPU::

    python eval_vsibench.py \
        --checkpoint_dir ../ckpt/DKVLM_Qwen3_8B \
        --base_model_path Qwen/Qwen3-VL-8B-Instruct \
        --geometry_ckpt weights/vggt_omega_1b_512.pt \
        --vsibench_dir ../data/VSI-Bench \
        --output_dir runs/vsibench

Eight GPUs (samples are sharded round-robin and merged on rank 0)::

    torchrun --nproc_per_node 8 eval_vsibench.py ...same flags...

Scoring follows the VSI-Bench protocol: exact match on multiple-choice questions and mean
relative accuracy over thresholds 0.5 to 0.95 on numeric ones. The headline ``overall`` is
the unweighted mean across question types, with the three relative-direction difficulties
averaged into one before that mean is taken.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")

import numpy as np

from dkvlm import load_dkvlm
from dkvlm.loader import build_prompt
from eval_utils import (
    fuzzy_matching,
    gather_results,
    mean_relative_accuracy,
    setup_distributed,
    to_float,
)

MCA_QUESTION_TYPES = {
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
}
NA_QUESTION_TYPES = {
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
}
MRA_METRIC = "MRA:.5:.95:.05"


def score_sample(item: Dict[str, Any], output_text: str) -> Tuple[str, float]:
    qtype = item.get("question_type")
    gt = item.get("ground_truth", "")
    pred = fuzzy_matching(output_text)
    if qtype in MCA_QUESTION_TYPES:
        return "accuracy", 1.0 if str(pred).lower() == str(gt).lower() else 0.0
    if qtype in NA_QUESTION_TYPES:
        pred_f = to_float(pred)
        gt_f = to_float(fuzzy_matching(gt))
        if pred_f is None or gt_f is None:
            return MRA_METRIC, 0.0
        return MRA_METRIC, mean_relative_accuracy(pred_f, gt_f)
    raise ValueError(f"unknown question type: {qtype}")


def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    acc_by_q: Dict[str, List[float]] = defaultdict(list)
    mra_by_q: Dict[str, List[float]] = defaultdict(list)
    for r in results:
        qtype = r.get("question_type")
        name = r.get("metric_name")
        val = r.get("metric_value")
        if val is None:
            continue
        if qtype in MCA_QUESTION_TYPES and name == "accuracy":
            acc_by_q[str(qtype)].append(float(val))
        elif qtype in NA_QUESTION_TYPES and name == MRA_METRIC:
            mra_by_q[str(qtype)].append(float(val))

    output: Dict[str, float] = {}
    for qtype, vals in acc_by_q.items():
        output[f"{qtype}_accuracy"] = float(np.nanmean(vals)) if vals else 0.0
    for qtype, vals in mra_by_q.items():
        output[f"{qtype}_{MRA_METRIC}"] = float(np.nanmean(vals)) if vals else 0.0

    difficulties = [
        f"object_rel_direction_{d}_accuracy" for d in ("easy", "medium", "hard")
    ]
    if all(k in output for k in difficulties):
        output["object_rel_direction_accuracy"] = (
            sum(output.pop(k) for k in difficulties) / 3.0
        )

    vals = [float(v) for v in output.values()]
    overall = float(np.nanmean(vals)) if vals else 0.0
    if np.isnan(overall):
        overall = 0.0
    return {"overall": overall, "overall_score_percent": overall * 100.0, **output}


def video_path_for_item(item: Dict[str, Any], roots: List[Path]) -> Path:
    rel = Path(str(item.get("dataset", "")).strip()) / f"{str(item.get('scene_name', '')).strip()}.mp4"
    for root in roots:
        candidate = (root / rel).resolve()
        if candidate.is_file():
            return candidate
    return (roots[0] / rel).resolve()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint_dir", required=True, help="DK-VLM checkpoint directory")
    ap.add_argument("--base_model_path", required=True, help="Qwen2.5-VL or Qwen3-VL base weights")
    ap.add_argument("--geometry_ckpt", default="", help="VGGT-Omega checkpoint (.pt)")
    ap.add_argument("--vsibench_dir", required=True, help="dir with test.jsonl and videos")
    ap.add_argument("--video_dir", default="", help="video root, if not --vsibench_dir")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--nframes", type=int, default=32)
    ap.add_argument("--max_samples", type=int, default=-1, help="-1 evaluates all samples")
    ap.add_argument(
        "--question_type",
        default="",
        help="if set, only evaluate this VSI question_type (e.g. route_planning)",
    )
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument(
        "--attn_implementation",
        default="flash_attention_2",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    ap.add_argument(
        "--geo_attention_backend",
        default="sdpa",
        choices=["eager", "sdpa", "chunked"],
    )
    ap.add_argument(
        "--scene_cache_size",
        type=int,
        default=32,
        help="Per-video cache of decoded frames + VGGT geometry (0 disables)",
    )
    ap.add_argument(
        "--no_geometry",
        action="store_true",
        help="disable the geometric branch while retaining DK-VLM's LoRA adapter",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    device, rank, world_size = setup_distributed()

    out_dir = Path(args.output_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    bench_dir = Path(args.vsibench_dir)
    video_roots = [Path(p) for p in (args.video_dir or str(bench_dir)).split(":") if p]
    items = [
        json.loads(line)
        for line in (bench_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    total = len(items)
    if str(args.question_type or "").strip():
        want = str(args.question_type).strip()
        items = [it for it in items if str(it.get("question_type") or "") == want]
    if args.max_samples > 0:
        items = items[: args.max_samples]

    use_geometry = not args.no_geometry
    model = load_dkvlm(
        args.checkpoint_dir,
        base_model_path=args.base_model_path,
        geometry_ckpt=args.geometry_ckpt,
        device=device,
        attn_implementation=args.attn_implementation,
        geo_attention_backend=args.geo_attention_backend,
        nframes=args.nframes,
        use_geometry=use_geometry,
        verbose=(rank == 0),
    )
    model.max_new_tokens = int(args.max_new_tokens)

    shard = items[rank::world_size]
    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    started = time.perf_counter()
    scene_cache: Dict[str, Tuple[Any, Any]] = {}
    scene_cache_order: List[str] = []
    cache_hits = 0
    cache_misses = 0

    def _cached_scene(video: Path, sample_id: str):
        nonlocal cache_hits, cache_misses
        key = str(video)
        if args.scene_cache_size > 0 and key in scene_cache:
            cache_hits += 1
            return scene_cache[key]
        sampled = model.resolve_frames(video)
        geometry = model.compute_geometry(sampled, sample_id) if use_geometry else None
        cache_misses += 1
        if args.scene_cache_size > 0:
            scene_cache[key] = (sampled, geometry)
            scene_cache_order.append(key)
            while len(scene_cache_order) > args.scene_cache_size:
                old = scene_cache_order.pop(0)
                scene_cache.pop(old, None)
        return sampled, geometry

    for i, item in enumerate(shard, start=1):
        try:
            video = video_path_for_item(item, video_roots)
            if not video.is_file():
                raise FileNotFoundError(f"video not found: {video}")
            sample_id = f"{item.get('dataset', '')}/{item.get('scene_name', '')}"
            sampled, geometry = _cached_scene(video, sample_id)
            prompt = build_prompt(item.get("question", ""), item.get("options"))
            inputs = model.build_inputs(
                sampled, prompt, geometry=geometry, sample_id=sample_id
            )
            output_text = model.generate(inputs)
            metric_name, metric_value = score_sample(item, output_text)
            results.append(
                {
                    "question_type": item.get("question_type"),
                    "question": item.get("question"),
                    "ground_truth": item.get("ground_truth"),
                    "prediction": output_text,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "video_path": str(video),
                }
            )
        except Exception as exc:
            failures.append({"sample_id": str(item.get("scene_name", "")), "error": repr(exc)})

        if rank == 0 and (i % 25 == 0 or i == len(shard)):
            elapsed = time.perf_counter() - started
            rate = i / max(elapsed, 1e-6)
            eta = (len(shard) - i) / max(rate, 1e-6)
            print(
                f"[vsibench] rank0 {i}/{len(shard)}  {rate:.2f} samples/s  eta {eta / 60:.1f} min",
                flush=True,
            )

    payload = {"rank": rank, "results": results, "failures": failures}
    gathered = gather_results(payload, world_size)

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
        full_dataset_evaluated = (
            evaluation_complete and len(items) == total
            and not str(args.question_type or "").strip()
        )
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
                "checkpoint_dir": str(args.checkpoint_dir),
                "base_model_path": str(args.base_model_path),
                "nframes": args.nframes,
                "max_new_tokens": args.max_new_tokens,
                "attn_implementation": args.attn_implementation,
                "geo_attention_backend": args.geo_attention_backend,
                "scene_cache_size": args.scene_cache_size,
                "scene_cache_hits": cache_hits,
                "scene_cache_misses": cache_misses,
                "elapsed_sec": time.perf_counter() - started,
                "geometry": "off" if args.no_geometry else "on",
                "question_type": str(args.question_type or "") or None,
            },
        }
        (out_dir / "predictions.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in all_results),
            encoding="utf-8",
        )
        (out_dir / "metrics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if all_failures:
            (out_dir / "failures.jsonl").write_text(
                "\n".join(json.dumps(f, ensure_ascii=False) for f in all_failures),
                encoding="utf-8",
            )
        print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
        elapsed = time.perf_counter() - started
        if not evaluation_complete:
            print(
                "[vsibench] WARNING: evaluation incomplete; scores cover successful "
                "samples only, not a full benchmark result."
            )
        elif not full_dataset_evaluated:
            print("[vsibench] NOTE: this is a subset evaluation, not a full benchmark result.")
        print(f"\n[vsibench] overall = {metrics['overall_score_percent']:.2f}%")
        print(f"[vsibench] scored {len(all_results)} samples, {len(all_failures)} failed")
        print(
            f"[vsibench] {elapsed:.1f}s  {len(all_results) / max(elapsed, 1e-6):.2f} samples/s  "
            f"scene_cache hit/miss={cache_hits}/{cache_misses}"
        )
        print(f"[vsibench] wrote {out_dir / 'metrics.json'}")

    if world_size > 1:
        import torch.distributed as dist

        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
