# Reference-Frame Composition (RC)

**1,599,560 examples with 2-4 steps** form the constructed pool. The paper uses **150K filtered/sampled two- and three-step examples**; this release is not that exact training subset.

| JSONL file | Examples |
| --- | ---: |
| `rc_vsi_sft_all_2to4.jsonl` | 18,756 |
| `rc_spar_sft_all_2to4.part1.jsonl` | 790,402 |
| `rc_spar_sft_all_2to4.part2.jsonl` | 790,402 |

`question_type` identifies route actions, turn angles, view-sequence turns, or headings. `conversations` contains the instruction and JSON-array answer; `video`, `image`, or `images` supplies relative media paths. Angle targets are clockwise degrees in `[0, 360)`; action targets are `straight`, `back`, `left`, or `right`.

SHA-256 checksums and counts are in [the manifest](rc_2to4_export_manifest.json). Original media are not included: obtain them through [VSI-590K](https://huggingface.co/datasets/nyu-visionx/VSI-590K) and [SPAR-7M](https://huggingface.co/datasets/jasonzhango/SPAR-7M), retaining the relative directory layout and source-dataset terms.

Original RC construction contributions use [MIT](../../LICENSE); source terms remain applicable. See [third-party notices](../../THIRD_PARTY.md).
