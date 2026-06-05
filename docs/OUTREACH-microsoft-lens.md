# Reaching the Microsoft Lens team

Use this as a template. The Lens repo is **inference-only** today — there is no training script upstream. A polite **GitHub issue** (show-and-tell + link) is the right first step, not an unsolicited large PR.

## Recommended channels (in order)

1. **Hugging Face discussion** on [microsoft/Lens](https://huggingface.co/microsoft/Lens/discussions) — see [OUTREACH-huggingface-lens-discussion.md](./OUTREACH-huggingface-lens-discussion.md)
2. **GitHub Issue** on [microsoft/Lens](https://github.com/microsoft/Lens/issues) — best fit for “community trainer exists”
3. **Hugging Face model card** — link your LoRA + trainer repo in the card README
4. **Optional:** comment on an existing training-related issue (e.g. #6) if relevant — do not spam every issue

Microsoft Lens does **not** appear to use GitHub Discussions. Pull requests are unlikely to be accepted unless they ask for training support.

## Draft issue title

```
Community: config-driven Lens-Base LoRA trainer (flow-match, Comfy export) — working on 24GB
```

## Draft issue body

```markdown
Hi Lens team,

We've built and validated a **standalone LoRA trainer** for **microsoft/Lens-Base** and wanted to share it in case it's useful to the community or future official tooling.

**Trainer repo:** https://github.com/LoboForge/LoboForge-LensTrainer  
**Hugging Face Space:** https://huggingface.co/spaces/LoboForge/LoboForge-LensTrainer

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
```

## Submit the issue

```bash
gh auth login   # once
bash scripts/open_microsoft_lens_issue.sh
```

Issue body: [OUTREACH-microsoft-lens-issue-body.md](./OUTREACH-microsoft-lens-issue-body.md)

**Opened:** https://github.com/microsoft/Lens/issues/12

## What to attach (builds credibility)

- Link to trainer repo + tagged release (`v0.1.0` or similar)
- One LoRA writeup with before/after or base vs LoRA samples
- Short note: GPU model, VRAM, steps, dataset size
- Optional: `loss.json` plot or final loss trend — keep it factual, not marketing

## What to avoid

- Opening a PR that adds the whole trainer into `microsoft/Lens` without maintainer interest
- Claiming Microsoft endorsement on Hugging Face
- Uploading GPT-OSS or Lens-Base weights — only your LoRA adapter

## Longer term

If the team is interested, a smaller upstream contribution could be:
- A `TRAINING.md` doc pointing to community trainers
- An official `examples/lora/` config snippet
- A link from the main README “Community tools” section

That usually follows a positive issue response, not before.
