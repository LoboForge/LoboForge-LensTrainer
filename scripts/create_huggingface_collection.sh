#!/usr/bin/env bash
# Create LoboForge HF collection: Lens-Base + Trainer Space + example LoRA.
set -euo pipefail

python3.12 <<'PY'
from huggingface_hub import HfApi

api = HfApi()
slug_hint = "lens-training-loboforge"
title = "Lens training (LoboForge)"
description = (
    "Microsoft Lens-Base, LoboForge LensTrainer, and community LoRA examples "
    "trained on Lens-Base."
)

col = api.create_collection(
    title=title,
    namespace="LoboForge",
    description=description,
    exists_ok=True,
)
print(f"Collection: {col.url}")

items = [
    ("microsoft/Lens-Base", "model", "Official Lens-Base weights"),
    ("LoboForge/LoboForge-LensTrainer", "space", "LensTrainer docs Space"),
    ("LoboForge/lens-lora-sebastian-jessica-v2", "model", "Sebastian + Jessica v2 LoRA"),
]
for item_id, item_type, note in items:
    api.add_collection_item(
        col.slug,
        item_id=item_id,
        item_type=item_type,
        note=note,
        exists_ok=True,
    )
    print(f"  + {item_type}: {item_id}")

print(f"\nDone: {col.url}")
PY
