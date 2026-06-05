# Published LoRAs

Example LoRAs trained with [LoboForge-LensTrainer](https://github.com/LoboForge/LoboForge-LensTrainer) on **microsoft/Lens-Base**.

Weights are **not** stored in this git repo (`.safetensors` are gitignored). Each writeup links to a Hugging Face model card or release asset.

| LoRA | Base model | Steps | Dataset | Writeup |
|------|------------|-------|---------|---------|
| Sebastian + Jessica v2 | Lens-Base | 5000 (published) | 24× 1024² dual-character | [HF](https://huggingface.co/LoboForge/lens-lora-sebastian-jessica-v2) · [writeup](./sebastian-jessica-v2.md) |

## Adding a new LoRA writeup

1. Upload weights and samples to Hugging Face manually (`hf upload` or the web UI).
2. Copy 2–4 preview PNGs into `docs/loras/assets/<name>/`.
3. Add `docs/loras/<name>.md` using `sebastian-jessica-v2.md` as a template.
4. Link it from this table and from the main README.
