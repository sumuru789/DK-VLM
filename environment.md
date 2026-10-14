# Environment Setup

## Dependencies

```text
Python 3.10.20
PyTorch 2.6.0+cu124
torchvision 0.21.0+cu124
CUDA 12.4
cuDNN 9.1.0
NCCL 2.21.5
transformers 4.57.6
tokenizers 0.22.2
accelerate 1.7.0
peft 0.17.1
safetensors 0.7.0
huggingface-hub 0.36.2
numpy 2.2.6
Pillow 12.2.0
decord 0.6.0
einops 0.8.2
flash-attn 2.7.4.post1
triton 3.2.0
qwen-vl-utils 0.0.14 (source on PYTHONPATH, not an installed package)
```

## Installation

Run the following commands from the repository root.

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install packaging ninja wheel
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation --no-deps

git clone https://github.com/facebookresearch/vggt-omega.git third_party/vggt-omega
git clone https://github.com/QwenLM/Qwen3-VL.git third_party/Qwen3-VL
git -C third_party/Qwen3-VL checkout 96588727e44c78b25ba03ea03b8e12f7e64fd0da
export VGGT_OMEGA_ROOT="$(pwd)/third_party/vggt-omega"
export PYTHONPATH="$(pwd)/third_party/Qwen3-VL/qwen-vl-utils/src${PYTHONPATH:+:$PYTHONPATH}"
export FORCE_QWENVL_VIDEO_READER=decord
```

The reported results were obtained on NVIDIA H200 GPUs.
