#!/usr/bin/env python3
"""Create a community post on microsoft/Lens Hugging Face discussions."""
from __future__ import annotations

import re
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "docs" / "OUTREACH-huggingface-lens-discussion.md"
REPO_ID = "microsoft/Lens"


def parse_draft(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    title_match = re.search(r"## Title\s+```\s*\n(.+?)\n```", text, re.DOTALL)
    body_match = re.search(r"## Body\s+```markdown\s*\n(.+?)\n```", text, re.DOTALL)
    if not title_match or not body_match:
        raise ValueError(f"Could not parse title/body from {path}")
    return title_match.group(1).strip(), body_match.group(1).strip()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print title and body only (use if API rate-limited)",
    )
    args = parser.parse_args()

    title, description = parse_draft(DRAFT)
    if args.dry_run:
        print(f"Post at: https://huggingface.co/{REPO_ID}/discussions/new\n")
        print(f"Title:\n{title}\n")
        print(f"Body:\n{description}\n")
        return 0

    api = HfApi()
    try:
        discussion = api.create_discussion(
            repo_id=REPO_ID,
            title=title,
            description=description,
            repo_type="model",
        )
    except Exception as exc:
        if "429" in str(exc) or "rate limit" in str(exc).lower():
            print(
                "Hugging Face rate-limited discussion creation for this account.\n"
                "Paste manually: https://huggingface.co/microsoft/Lens/discussions/new\n"
                "Or retry later: python3.12 scripts/post_huggingface_lens_discussion.py\n",
                file=sys.stderr,
            )
            print(f"Title:\n{title}\n")
            print(f"Body:\n{description}\n")
            return 1
        raise
    url = f"https://huggingface.co/{REPO_ID}/discussions/{discussion.num}"
    print(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
