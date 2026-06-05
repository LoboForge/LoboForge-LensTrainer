---
license: other
license_name: polyform-noncommercial-1.0.0
license_link: https://polyformproject.org/licenses/noncommercial/1.0.0
base_model: microsoft/Lens-Base
tags:
  - lora
  - lens
  - comfyui
  - text-to-image
  - loboForge
pipeline_tag: text-to-image
---

# lens-lora-sebastian-jessica-v2

Dual-character LoRA for **[microsoft/Lens-Base](https://huggingface.co/microsoft/Lens-Base)** — Sebastian (blonde man, blue eyes, tuxedo) and Jessica (redhead, green eyes, purple dress).

Trained with **[LoboForge-LensTrainer](https://github.com/LoboForge/LoboForge-LensTrainer)** · [Trainer Space](https://huggingface.co/spaces/LoboForge/LoboForge-LensTrainer)

## Download

| File | Description |
|------|-------------|
| `lora_final.safetensors` | Final weights (8000 steps) |
| `config.resolved.json` | Resolved training config |

## Training

| Field | Value |
|-------|-------|
| Dataset | 24 image/caption pairs, 1024×1024 |
| Steps | 8000 |
| LoRA rank / alpha | 16 / 16 |
| Optimizer | AdamW 8-bit |
| Preset | `train_lora_dual_character_24gb` |

## Prompt format

Full-sentence captions — no single trigger token. Describe both characters and the scene, matching your training captions.

Example:

```
Sebastian is a blonde man with long hair and irish blue eyes in a tuxedo. Jessica is a redhead woman with long copper colored red hair and irish green eyes in a purple dress named. They are standing together facing forward and laughing
```

## Inference (Lens-Base)

| Setting | Value |
|---------|-------|
| Steps | 50 |
| CFG | 5.0 |
| Resolution | 1024×1024 |

ComfyUI: load **Lens-Base** + this LoRA (`diffusion_model.*` keys).

## Samples

![Standing, laughing](samples/step_008000_lora_standing_forward_laughing.png)

![Beach, holding hands](samples/step_008000_lora_beach_holding_hands.png)

## License

PolyForm Noncommercial License 1.0.0. Base model weights remain under the Microsoft / Hugging Face license for Lens-Base.
