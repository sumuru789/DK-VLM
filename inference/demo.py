#!/usr/bin/env python3
"""Ask DK-VLM a question about a single video.

    python demo.py \
        --checkpoint_dir ../ckpt/DKVLM_Qwen25_7B \
        --base_model_path Qwen/Qwen2.5-VL-7B-Instruct \
        --geometry_ckpt weights/vggt_omega_1b_512.pt \
        --video ../demo/sample.mp4 \
        --question "How many chair(s) are in this room?"

``--video`` also accepts a directory of frames, which are used in sorted filename order.
For multiple-choice questions pass ``--option`` once per choice; the prompt then asks for
a bare option letter instead of a word or phrase.

``--no_geometry`` disables the geometric branch but still loads DK-VLM's LoRA adapter.
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")

from dkvlm import load_dkvlm


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--base_model_path", required=True)
    ap.add_argument("--geometry_ckpt", default="")
    ap.add_argument("--video", required=True, help="video file or directory of frames")
    ap.add_argument("--question", required=True)
    ap.add_argument(
        "--option",
        action="append",
        default=None,
        metavar="TEXT",
        help="a multiple-choice option, e.g. --option 'A. left'; repeatable",
    )
    ap.add_argument("--nframes", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--attn_implementation", default="flash_attention_2", choices=["eager", "sdpa", "flash_attention_2"])
    ap.add_argument("--geo_attention_backend", default="sdpa", choices=["eager", "sdpa", "chunked"])
    ap.add_argument(
        "--no_geometry",
        action="store_true",
        help="disable the geometric branch while retaining DK-VLM's LoRA adapter",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    use_geometry = not args.no_geometry

    model = load_dkvlm(
        args.checkpoint_dir,
        base_model_path=args.base_model_path,
        geometry_ckpt=args.geometry_ckpt,
        attn_implementation=args.attn_implementation,
        geo_attention_backend=args.geo_attention_backend,
        nframes=args.nframes,
        use_geometry=use_geometry,
    )
    model.max_new_tokens = int(args.max_new_tokens)

    sampled = model.resolve_frames(args.video)
    print(f"[demo] sampled {len(sampled)} frames from {args.video}")

    from dkvlm.loader import build_prompt

    prompt = build_prompt(args.question, args.option)

    if use_geometry:
        t0 = time.perf_counter()
        geometry = model.compute_geometry(sampled, sample_id="demo")
        print(
            f"[demo] geometry: points {geometry.points.shape}, "
            f"poses {geometry.camera_poses.shape}  ({time.perf_counter() - t0:.1f}s)"
        )
    else:
        geometry = None

    inputs = model.build_inputs(sampled, prompt, geometry=geometry, sample_id="demo")
    if geometry is not None:
        stats = model.last_geometry_stats
        print(
            f"[demo] geometry covers {stats['joint_valid']}/{stats['visual']} visual tokens "
            f"({stats['valid_ratio'] * 100:.1f}%)"
        )
    answer = model.generate(inputs)
    print(f"\nQ: {args.question}")
    print(f"A: {answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
