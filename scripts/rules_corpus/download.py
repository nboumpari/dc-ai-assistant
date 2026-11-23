"""Download the rules agent's documents from their official sources.

Run from the project root:

    uv run python -m scripts.rules_corpus.download

Files go to data/documents/raw/ (gitignored). Files already present are kept,
so the script is safe to run again. Documents that cannot be downloaded
automatically (ADR) are reported with instructions for a manual download.
A manifest with size and SHA-256 of every file is written next to them.
"""

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name("corpus.yaml")
RAW_DIR = ROOT / "data" / "documents" / "raw"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) dc-ai-assistant corpus downloader"


def load_corpus() -> list[dict]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["documents"]


def _looks_right(data: bytes, accept: str) -> bool:
    if "pdf" in accept:
        return data.startswith(b"%PDF")
    return b"<html" in data[:2000].lower()


def download(url: str, target: Path, accept: str = "application/pdf", attempts: int = 3) -> None:
    """Fetch one file; retry briefly if the server answers with an empty or wrong body."""
    headers = {"User-Agent": USER_AGENT, "Accept": accept, "Accept-Language": "eng"}
    for attempt in range(1, attempts + 1):
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
            data = response.read()
        if _looks_right(data, accept):
            target.write_bytes(data)
            return
        time.sleep(5 * attempt)
    raise ValueError(f"{url} did not return {accept} after {attempts} attempts")


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest, missing = {}, []
    for doc in load_corpus():
        for name in doc["files"]:
            path = RAW_DIR / name
            if not path.exists():
                if doc.get("manual_download"):
                    missing.append((doc, name))
                    continue
                print(f"downloading {name} ...")
                download(doc.get("download_url", doc["url"]), path, doc.get("accept", "application/pdf"))
            data = path.read_bytes()
            manifest[name] = {
                "document": doc["id"],
                "source": doc["url"],
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            print(f"  ok  {name:<36} {len(data) / 1e6:6.1f} MB")

    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if missing:
        print("\nManual download needed:")
        for doc, name in missing:
            print(f"  - {name}: {doc['manual_download']}")
        return 1
    print(f"\nAll {len(manifest)} files present in {RAW_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
