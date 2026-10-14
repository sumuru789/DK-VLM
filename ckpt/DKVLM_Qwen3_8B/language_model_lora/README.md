# DK-VLM-8B LoRA adapter

Base: [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct).

This is an incremental adapter, not a full model. Load its parent checkpoint directory, including `geo_branch_state.pt` and `dkvlm_config.json`, with `dkvlm.load_dkvlm`; supply the base model and `vggt_omega_1b_512.pt` separately. See [inference instructions](../../../README.md#quick-start) and [license notices](../../../THIRD_PARTY.md).
