# Sebastian + Jessica v2 — Dual Character LoRA (Microsoft Lens)

LoRA fine-tune for **Microsoft Lens** — two recurring characters:

- **Sebastian** — blonde man, long hair, Irish blue eyes, tuxedo
- **Jessica** — redhead, long copper-red hair, Irish green eyes, purple dress

Trained on **Lens-Base** ([Hugging Face](https://huggingface.co/microsoft/Lens-Base)) with [LoboForge-LensTrainer](https://github.com/LoboForge/LoboForge-LensTrainer).

Also on Hugging Face: [LoboForge/lens-lora-sebastian-jessica-v2](https://huggingface.co/LoboForge/lens-lora-sebastian-jessica-v2)

## Training

| Field | Value |
|-------|-------|
| Dataset | 24 image/caption pairs, 1024×1024 |
| Steps | 8000 |
| LoRA rank / alpha | 16 / 16 |
| Optimizer | AdamW 8-bit |

## Prompt format

**Full-sentence captions** — no single trigger token. Describe both characters and the scene, matching your training captions.

Example:

```
Sebastian is a blonde man with long hair and irish blue eyes in a tuxedo. Jessica is a redhead woman with long copper colored red hair and irish green eyes in a purple dress named. They are standing together facing forward and laughing
```

## Inference

| Setting | Value |
|---------|-------|
| Base | Lens or Lens-Base (HF) |
| Steps | 50 |
| CFG | 5.0 |
| Resolution | 1024×1024 |

ComfyUI: load Lens checkpoint + this LoRA (`diffusion_model.*` keys).

## License

PolyForm Noncommercial License 1.0.0 — non-commercial use only.
