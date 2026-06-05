#!/usr/bin/env python3
"""Publish Sebastian + Jessica v2 LoRA to CivitAI (civitai.com)."""
from __future__ import annotations

import argparse
import logging
import math
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "output" / "lens-lora-sebastian-jessica-v2"
DEFAULT_CARD = ROOT / "huggingface" / "lens-lora-sebastian-jessica-v2" / "civitai.md"
DEFAULT_SAMPLES = ROOT / "docs" / "loras" / "assets" / "sebastian-jessica-v2"

MODEL_NAME = "Sebastian + Jessica v2 — Dual Character LoRA (Lens)"
VERSION_NAME = "v2.0"
BASE_MODEL = "Lens"
# CivitAI mirror: microsoft/Lens "Lens" checkpoint (closest to Lens-Base on HF)
RECOMMENDED_LENS_VERSION_ID = 2982236

SAMPLE_IMAGES = (
    "step_008000_lora_standing_forward_laughing.png",
    "step_008000_lora_beach_holding_hands.png",
)


def load_token(explicit: str | None) -> str:
    if explicit:
        return explicit.strip()
    for key in ("CIVITAI_API_TOKEN", "CIVITAI_TOKEN"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    env_file = ROOT / "training.env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip() in ("CIVITAI_API_TOKEN", "CIVITAI_TOKEN"):
                return value.strip().strip('"').strip("'")
    sys.exit(
        "Missing CivitAI token. Set CIVITAI_API_TOKEN, add it to training.env, "
        "or pass --token."
    )


def civitai_client(token: str):
    from civitai.client import CivitAIClient
    from civitai.session.whoami import get_whoami_by_raw_user_info

    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Referer": "https://civitai.com/",
        }
    )
    me = session.get("https://civitai.com/api/v1/me", timeout=60).json()
    whoami = get_whoami_by_raw_user_info(
        {
            "id": me["id"],
            "username": me["username"],
            "email": me.get("email", ""),
            "createdAt": "2020-01-01T00:00:00Z",
            "name": me.get("username", "user"),
        }
    )
    return CivitAIClient(session=session, raw_user_info=whoami.raw if whoami else None)


def upload_model_file(session: requests.Session, local_file: str, remote_name: str) -> dict:
    """Upload a LoRA file, using multipart + curl when CivitAI splits large files."""
    size = os.path.getsize(local_file)
    logging.info("Uploading %s (%d MB) ...", remote_name, size // (1024 * 1024))
    init = session.post(
        "https://civitai.com/api/upload",
        json={"filename": remote_name, "type": "model", "size": size},
        headers={"Referer": "https://civitai.com/models/0/wizard?step=3"},
        timeout=120,
    )
    init.raise_for_status()
    upload_data = init.json()
    parts = upload_data["urls"]
    referer = "https://civitai.com/models/0/wizard?step=3"

    if len(parts) == 1:
        with open(local_file, "rb") as handle:
            resp = session.put(
                parts[0]["url"],
                data=handle,
                headers={"Referer": referer, "Content-Length": str(size)},
                timeout=3600,
            )
        resp.raise_for_status()
        etag = resp.headers["ETag"]
    else:
        chunk_size = math.ceil(size / len(parts))
        uploaded_parts = []
        with open(local_file, "rb") as handle, tempfile.TemporaryDirectory() as tmp:
            for index, part in enumerate(parts, 1):
                chunk = handle.read(chunk_size)
                part_path = os.path.join(tmp, f"part_{index}.bin")
                with open(part_path, "wb") as part_file:
                    part_file.write(chunk)
                logging.info("  part %d/%d (%d MB)", index, len(parts), len(chunk) // (1024 * 1024))
                header_path = os.path.join(tmp, f"part_{index}.hdr")
                subprocess.run(
                    [
                        "curl",
                        "-sS",
                        "-X",
                        "PUT",
                        part["url"],
                        "--http1.1",
                        "-H",
                        f"Content-Length: {len(chunk)}",
                        "--data-binary",
                        f"@{part_path}",
                        "-D",
                        header_path,
                        "-o",
                        os.devnull,
                    ],
                    check=True,
                    timeout=3600,
                )
                etag = None
                for line in Path(header_path).read_text().splitlines():
                    if line.lower().startswith("etag:"):
                        etag = line.split(":", 1)[1].strip()
                if not etag:
                    raise RuntimeError(f"Missing ETag for upload part {index}")
                uploaded_parts.append({"ETag": etag, "PartNumber": int(part["partNumber"])})

        complete = session.post(
            "https://civitai.com/api/upload/complete",
            json={
                "bucket": upload_data["bucket"],
                "key": upload_data["key"],
                "type": "model",
                "uploadId": upload_data["uploadId"],
                "parts": uploaded_parts,
                "backend": upload_data.get("backend"),
            },
            headers={"Referer": referer},
            timeout=120,
        )
        complete.raise_for_status()
        etag = complete.text.strip().strip('"')

    return {
        "url": parts[0]["url"].split("?", 1)[0],
        "bucket": upload_data["bucket"],
        "key": upload_data["key"],
        "name": remote_name,
        "uuid": str(uuid.uuid4()),
        "sizeKB": size / 1024.0,
    }


def attach_model_file(client, model_version_id: int, local_file: str, remote_name: str) -> None:
    upload = upload_model_file(client._session, local_file, remote_name)
    client._post(
        "/api/trpc/modelFile.create",
        {
            "url": upload["url"],
            "bucket": upload["bucket"],
            "key": upload["key"],
            "name": remote_name,
            "uuid": upload["uuid"],
            "sizeKB": upload["sizeKB"],
            "modelVersionId": model_version_id,
            "type": "Model",
            "metadata": {"size": None, "fp": None},
            "authed": True,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", help="CivitAI API token (or use CIVITAI_API_TOKEN / training.env)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--card", type=Path, default=DEFAULT_CARD)
    parser.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and token only")
    parser.add_argument("--model-id", type=int, help="Resume an existing CivitAI model draft")
    parser.add_argument("--version-id", type=int, help="Resume an existing CivitAI version draft")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="==> %(message)s")

    token = load_token(args.token)
    weights = args.output_dir / "lora_final.safetensors"
    if not weights.is_file():
        logging.error("Missing weights: %s", weights)
        return 1
    if not args.card.is_file():
        logging.error("Missing card: %s", args.card)
        return 1

    sample_paths = []
    for name in SAMPLE_IMAGES:
        path = args.samples_dir / name
        if not path.is_file():
            path = args.output_dir / "samples" / name
        if not path.is_file():
            logging.error("Missing sample image: %s", name)
            return 1
        sample_paths.append(str(path))

    client = civitai_client(token)
    logging.info("CivitAI account: %s", client.whoami.username)

    if args.dry_run:
        logging.info("Dry run OK — weights, card, samples, and token verified.")
        return 0

    description = args.card.read_text(encoding="utf-8")
    tags = [
        "character",
        "lens",
        "microsoft lens",
        "dual character",
        "sebastian",
        "jessica",
        "comfyui",
        "text-to-image",
    ]

    model_kwargs = dict(
        name=MODEL_NAME,
        description_md=description,
        tags=tags,
        category="character",
        type_="LORA",
        commercial_use=[],
        allow_no_credit=False,
        allow_derivatives=False,
        allow_different_licence=False,
        nsfw=False,
        poi=False,
    )
    if args.model_id:
        model_kwargs["exist_model_id"] = args.model_id
    model = client.upsert_model(**model_kwargs)
    model_id = model["id"]
    logging.info("Model id: %s", model_id)

    version_kwargs = dict(
        model_id=model_id,
        version_name=VERSION_NAME,
        description_md=description,
        trigger_words=[],
        base_model=BASE_MODEL,
        steps=8000,
        epochs=None,
        clip_skip=None,
        recommended_resources=[RECOMMENDED_LENS_VERSION_ID],
        require_auth_when_download=False,
    )
    if args.version_id:
        version_kwargs["exist_version_id"] = args.version_id
    version = client.upsert_version(**version_kwargs)
    version_id = version["id"]
    logging.info("Version id: %s", version_id)

    attach_model_file(
        client,
        version_id,
        str(weights),
        "lens-lora-sebastian-jessica-v2.safetensors",
    )

    post_id = client.upload_images_for_model_version(
        model_version_id=version_id,
        image_files=sample_paths,
        tags=["character", "lens", "sebastian", "jessica"],
        nsfw=False,
    )
    logging.info("Gallery post id: %s", post_id)
    client.post_publish(post_id)

    client.model_publish(model_id=model_id, model_version_id=version_id)
    url = f"https://civitai.com/models/{model_id}"
    logging.info("Published: %s", url)
    print(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
