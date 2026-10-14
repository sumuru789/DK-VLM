# DK-VLM inference and evaluation

See the [release README](../README.md) for setup, checkpoints, and single-video inference.

## Evaluation

Complete the [environment setup](../environment.md) first, then prepare the datasets below. Run the download commands from the repository root.

### Data Preparation

**VSI-Bench.** Download the original full benchmark from [Hugging Face](https://huggingface.co/datasets/nyu-visionx/VSI-Bench) and extract the three video archives:

```bash
hf download nyu-visionx/VSI-Bench test.jsonl arkitscenes.zip scannet.zip scannetpp.zip \
  --repo-type dataset --local-dir data/VSI-Bench
for archive in arkitscenes scannet scannetpp; do
  unzip "data/VSI-Bench/${archive}.zip" -d data/VSI-Bench
done
```

**ReVSI-32.** Download the official [32-frame annotations and videos](https://huggingface.co/datasets/3dlg-hcvc/ReVSI), extract the 32-frame videos, and convert the annotations to JSONL:

```bash
hf download 3dlg-hcvc/ReVSI 32_frame/test-00000-of-00001.parquet video.zip \
  --repo-type dataset --local-dir data/ReVSI
unzip data/ReVSI/video.zip '32_frame/*' -d data/ReVSI
python -m pip install pyarrow
python inference/parquet_to_jsonl.py \
  data/ReVSI/32_frame/test-00000-of-00001.parquet data/ReVSI/32_frame/test.jsonl
```

The resulting layout should be:

```text
data/
├── VSI-Bench/
│   ├── test.jsonl
│   ├── arkitscenes/*.mp4
│   ├── scannet/*.mp4
│   └── scannetpp/*.mp4
└── ReVSI/
    └── 32_frame/
        ├── test.jsonl
        └── *.mp4
```

### Run Evaluation

Enter the evaluation directory:

```bash
cd inference
```

Replace `/path/to/...` with local paths. For 7B, switch to `../ckpt/DKVLM_Qwen25_7B` and the matching Qwen2.5-VL base model. For single-GPU evaluation, replace `torchrun --standalone --nproc_per_node=8` with `python`.

#### VSI-Bench

The benchmark directory should contain `test.jsonl` and videos at `<dataset>/<scene_name>.mp4`.

```bash
torchrun --standalone --nproc_per_node=8 eval_vsibench.py \
  --checkpoint_dir ../ckpt/DKVLM_Qwen3_8B \
  --base_model_path /path/to/Qwen3-VL-8B-Instruct \
  --geometry_ckpt /path/to/vggt_omega_1b_512.pt \
  --vsibench_dir ../data/VSI-Bench \
  --output_dir runs/vsibench
```

#### ReVSI-32

The benchmark directory should contain `32_frame/test.jsonl` and videos at `32_frame/<scene_id>.mp4`.

```bash
torchrun --standalone --nproc_per_node=8 eval_revsi.py \
  --checkpoint_dir ../ckpt/DKVLM_Qwen3_8B \
  --base_model_path /path/to/Qwen3-VL-8B-Instruct \
  --geometry_ckpt /path/to/vggt_omega_1b_512.pt \
  --revsi_dir ../data/ReVSI \
  --output_dir runs/revsi
```

