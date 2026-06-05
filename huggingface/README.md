---
title: LoboForge LensTrainer
emoji: 🐺
colorFrom: blue
colorTo: purple
sdk: static
pinned: true
license: other
license_name: polyform-noncommercial-1.0.0
license_link: https://polyformproject.org/licenses/noncommercial/1.0.0
short_description: LoRA trainer for Microsoft Lens-Base
---

# LoboForge LensTrainer

Train ComfyUI-compatible LoRAs for **[microsoft/Lens-Base](https://huggingface.co/microsoft/Lens-Base)**.

**GitHub:** https://github.com/LoboForge/LoboForge-LensTrainer

This Space is a **documentation landing page**. Training runs on your local GPU or RunPod — not in the browser.

## Features

- `python train.py configs/train_lora_lens_base_24gb.yaml`
- Flow-match loss matching `LensPipeline`
- PEFT LoRA on `LensTransformer2DModel`; Comfy `diffusion_model.*` export
- 16–24GB VRAM presets

## Example LoRA

[Sebastian + Jessica v2](https://huggingface.co/LoboForge/lens-lora-sebastian-jessica-v2)
