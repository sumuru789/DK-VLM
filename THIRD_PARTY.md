# Third-party notices

MIT covers original DK-VLM contributions only. It does not relicense upstream material or remove its conditions. Original incremental weights and newly constructed RC annotations are offered under MIT subject to applicable upstream terms; base models, frontend weights, and full source media datasets are not redistributed here. The bundled demo video retains its original license, as noted below.

| Component/source | Upstream terms |
| --- | --- |
| [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), and [Transformers](https://github.com/huggingface/transformers) | Apache 2.0; upstream attention interfaces and any adapted portions retain those terms. A copy is in [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt). |
| [VGGT-Omega](https://github.com/facebookresearch/vggt-omega) | FAIR Noncommercial Research License. Geometry preprocessing in `inference/dkvlm/geometry.py` adapts the upstream resize/crop/pad procedure; those portions retain this license. A copy is in [licenses/VGGT-Omega.txt](licenses/VGGT-Omega.txt). |
| [VSI-590K](https://huggingface.co/datasets/nyu-visionx/VSI-590K) | Source annotations are listed as Apache 2.0. Derived source portions retain upstream attribution and terms. |
| [SPAR-7M](https://huggingface.co/datasets/jasonzhango/SPAR-7M) | Upstream code and annotations use MIT. Original media follow the respective source-dataset terms. |
| [ARKitScenes demo video](demo/README.md) | `demo/sample.mp4` is an unmodified ARKitScenes video distributed through VSI-Bench. It retains the [Apple license and copyright notice](demo/ARKitScenes-LICENSE.txt), not MIT. |

VGGT-Omega preprocessing portions: Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved. DK-VLM adds pixel-transform metadata and token-alignment handling.

Users must obtain raw media from the original datasets and comply with their terms. An MIT label for our code or RC construction contributions does not grant rights to those images/videos or make the VGGT-Omega-based pipeline unrestricted for commercial use.
