# Publishing on Hugging Face

Hugging Face is not one upload — you typically publish **three related things**:

| Asset | HF type | Purpose |
|-------|---------|---------|
| **Trainer** (this repo) | **Space** | Discovery, docs, links to GitHub |
| **LoRA weights** | **Model** repo | Downloadable `.safetensors` |
| **Optional** | **Collection** | Groups Base + Trainer + LoRAs |

Weights and code use different repo types. Do **not** put the full trainer codebase in a Model repo.

---

## 1. Publish the trainer (Space)

A **Space** is the standard way to list a tool on Hugging Face. This repo includes a **static** landing page in `huggingface/` (docs only — no GPU training in the Space).

### Publish script

```bash
cd /path/to/LoboForge-LensTrainer
hf auth login

bash scripts/publish_huggingface_space.sh
```

Your Space URL: **https://huggingface.co/spaces/LoboForge/LoboForge-LensTrainer**

Uses `hf upload` (not git push) so your HF token works without separate git credentials.

### Space README

HF displays `huggingface/README.md` (YAML frontmatter + summary). The publish script uploads it as `README.md` plus `index.html`.

---

## 2. Publish LoRA weights (Model repo)

```bash
bash scripts/publish_huggingface_lora.sh
```

Model card template: `huggingface/lens-lora-sebastian-jessica-v2/README.md`  
Writeup in git: [docs/loras/sebastian-jessica-v2.md](./loras/sebastian-jessica-v2.md)

---

## 3. Collection (recommended)

Group everything for users browsing the Hub:

```bash
bash scripts/create_huggingface_collection.sh
```

Or manually: https://huggingface.co/collections/new → title **Lens training (LoboForge)** → add:

- [microsoft/Lens-Base](https://huggingface.co/microsoft/Lens-Base)
- Space: `LoboForge/LoboForge-LensTrainer`
- Model: `LoboForge/lens-lora-sebastian-jessica-v2`

New HF accounts may hit a collection creation rate limit; retry the script later or use the manual link.

Link the collection from your org profile and GitHub README.

---

## 4. GitHub repo (source of truth)

Keep **full trainer code on GitHub** (`LoboForge/LoboForge-LensTrainer`). The Space points here for install/train.

In GitHub repo **About** (right sidebar):

- Website: `https://huggingface.co/spaces/LoboForge/LoboForge-LensTrainer`
- Topics: `lens`, `lora`, `diffusers`, `comfyui`, `microsoft-lens`

---

## 5. Tell Microsoft Lens

After the Space is live:

- Post on [microsoft/Lens discussions](https://huggingface.co/microsoft/Lens/discussions) — draft/script: [OUTREACH-huggingface-lens-discussion.md](./OUTREACH-huggingface-lens-discussion.md)
- Optional GitHub issue on [microsoft/Lens](https://github.com/microsoft/Lens/issues) using [OUTREACH-microsoft-lens.md](./OUTREACH-microsoft-lens.md)

Include:

- Trainer Space URL
- GitHub URL
- Example LoRA model URL

---

## Checklist

- [x] Space live: `spaces/LoboForge/LoboForge-LensTrainer`
- [x] LoRA model: `LoboForge/lens-lora-sebastian-jessica-v2`
- [ ] Collection created (`bash scripts/create_huggingface_collection.sh`)
- [ ] GitHub README links to Space
- [ ] `docs/loras/sebastian-jessica-v2.md` HF URL filled in
- [x] Issue on `microsoft/Lens`: https://github.com/microsoft/Lens/issues/12
