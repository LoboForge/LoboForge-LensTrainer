# Hugging Face discussion — microsoft/Lens

Post on: https://huggingface.co/microsoft/Lens/discussions

## Title

```
Community LoRA trainer for Lens-Base — now available
```

## Body

```markdown
Hi everyone,

We've finished building a **config-driven LoRA trainer for [microsoft/Lens-Base](https://huggingface.co/microsoft/Lens-Base)** and wanted to share it here in case it saves others the work of wiring up training from the inference codebase alone.

**GitHub (source):** https://github.com/LoboForge/LoboForge-LensTrainer  
**Hugging Face (docs Space):** https://huggingface.co/spaces/LoboForge/LoboForge-LensTrainer

### What it does

- `python train.py` with YAML presets (16–24GB VRAM options)
- Flow-match loss aligned with `LensPipeline` (GPT-OSS text features, Lens transformer latents)
- PEFT LoRA on `LensTransformer2DModel`; ComfyUI-compatible export (`diffusion_model.*` keys)
- Checkpoint resume, mid-training samples, `loss.json`

### Example output (community — not Microsoft-endorsed)

- LoRA weights: https://huggingface.co/LoboForge/lens-lora-sebastian-jessica-v2
- Training writeup: https://github.com/LoboForge/LoboForge-LensTrainer/blob/main/docs/loras/sebastian-jessica-v2.md

The trainer vendors the public [microsoft/Lens](https://github.com/microsoft/Lens) inference package and does not redistribute base weights.

Happy to answer questions on configs, VRAM presets, or Comfy export. Thanks to the Lens team for open-sourcing the model and inference stack.
```

## Post

**Manual (if API rate-limited):** log in as LoboForge → https://huggingface.co/microsoft/Lens/discussions/new → paste title + body above.

**API:**

```bash
python3.12 scripts/post_huggingface_lens_discussion.py
python3.12 scripts/post_huggingface_lens_discussion.py --dry-run   # print copy-paste content
```

New HF accounts may hit a discussion creation rate limit; use the manual link or retry later.
