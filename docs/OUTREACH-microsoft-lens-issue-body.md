Hi Lens team,

We've built and validated a **standalone LoRA trainer** for **microsoft/Lens-Base** and wanted to share it in case it's useful to the community or future official tooling.

**Trainer repo:** https://github.com/LoboForge/LoboForge-LensTrainer  
**Hugging Face Space:** https://huggingface.co/spaces/LoboForge/LoboForge-LensTrainer  
**HF collection:** https://huggingface.co/collections/LoboForge/lens-training-loboforge

**What works today**
- Config-driven training: `python train.py configs/train_lora_lens_base_24gb.yaml`
- Flow-match loss aligned with `LensPipeline` (latents `[B, H×W, 128]`, GPT-OSS multi-layer text, `timestep/1000`)
- PEFT LoRA on `LensTransformer2DModel`; TE/VAE frozen
- ComfyUI-compatible export (`diffusion_model.*` key remap)
- 24GB preset: CPU offload, TE + latent cache, `disable_mxfp4`, grad checkpointing, AdamW 8-bit
- Checkpoint resume, mid-training samples, `loss.json`

**Example result (community LoRA, not Microsoft-endorsed)**
- Writeup: https://github.com/LoboForge/LoboForge-LensTrainer/blob/main/docs/loras/sebastian-jessica-v2.md
- Weights (HF): https://huggingface.co/LoboForge/lens-lora-sebastian-jessica-v2

**Environment**
- Lens installed from `vendor/Lens` (git clone; upstream has no `pyproject.toml`)
- Tested on consumer/datacenter GPUs with Lens-Base local or HF hub

We're **not** asking for an immediate merge — mainly flagging that Lens-Base LoRA training is practical with the public inference package, and happy to answer questions or share configs/logs if helpful.

Thanks for open-sourcing Lens!
