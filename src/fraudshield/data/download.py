"""Download raw datasets into data/raw/<source>/ and record provenance.

Each source goes into its own directory. Archives are extracted with path checks,
and a provenance manifest (URL, licence, SHA-256, download time) is written to
data/raw/MANIFEST.json so results can be traced back to exact files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

SOURCES = {
    "uci_sms": {
        "url": "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
        "licence": "CC BY 4.0",
        "citation": "Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection. UCI ML Repository.",
        "sha256": None,
    },
    "mendeley_smishing": {
        "url": (
            "https://data.mendeley.com/public-files/datasets/f45bkkt8pr/files/"
            "edb361de-918d-469f-9106-e84823830665/file_downloaded"
        ),
        "licence": "CC BY 4.0",
        "citation": (
            "Mishra, S. & Soni, D. (2022). SMS Phishing Dataset for Machine Learning and "
            "Pattern Recognition. Mendeley Data, V1. doi:10.17632/f45bkkt8pr.1"
        ),
        # Published by Mendeley's file API.
        "sha256": "9bbf3188fdad81495d8e82825648b9b63b53fc86841a3d26c02629990b233cc3",
    },
    "phiusiil": {
        "url": "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip",
        "licence": "CC BY 4.0",
        "citation": (
            "Prasad, A. & Chandra, S. (2024). PhiUSIIL Phishing URL Dataset. UCI ML Repository. "
            "doi:10.24432/C5BC8M"
        ),
        "sha256": None,
    },
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_with_resume(url: str, dest: Path, fresh: bool = False, retries: int = 30) -> None:
    """Download to `dest.part`, resuming with HTTP Range after timeouts; rename when complete.

    A partial file is never left at `dest`, so an interrupted run can't be mistaken
    for a cached download.
    """
    part = dest.with_suffix(dest.suffix + ".part")
    if fresh and part.exists():
        part.unlink()
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "fraudshield-research/0.1"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                if have and resp.status != 206:  # server ignored Range: start over
                    have = 0
                total = resp.headers.get("Content-Length")
                total = int(total) + have if total else None
                with part.open("ab" if have else "wb") as f:
                    shutil.copyfileobj(resp, f, length=1 << 16)
            size = part.stat().st_size
            complete = size >= total if total else True
            # UCI sends no Content-Length, so also check the archive is intact.
            if complete and (dest.suffix != ".zip" or zipfile.is_zipfile(part)):
                part.rename(dest)
                return
            print(f"[download]   incomplete archive ({size} bytes), retrying", flush=True)
            part.unlink()
        except (TimeoutError, OSError) as exc:  # URLError subclasses OSError
            size = part.stat().st_size if part.exists() else 0
            print(f"[download]   attempt {attempt}: {type(exc).__name__} at {size} bytes, resuming",
                  flush=True)
    raise RuntimeError(f"Download failed after {retries} attempts: {url}")


def safe_extract(zip_path: Path, dest: Path) -> list[str]:
    """Extract a zip, refusing members that would escape `dest`."""
    dest = dest.resolve()
    names = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if not target.is_relative_to(dest):
                raise ValueError(f"Unsafe path in archive: {member.filename}")
            if member.is_dir():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
            names.append(member.filename)
    return names


def download(name: str, raw_dir: Path, force: bool = False) -> dict:
    spec = SOURCES[name]
    out_dir = raw_dir / name
    archive = out_dir / f"{name}.zip"
    out_dir.mkdir(parents=True, exist_ok=True)

    if force or not archive.exists():
        print(f"[download] {name} <- {spec['url']}", flush=True)
        fetch_with_resume(spec["url"], archive, fresh=force)
    else:
        print(f"[download] {name}: cached")

    digest = sha256_of(archive)
    if spec["sha256"] and digest != spec["sha256"]:
        raise ValueError(f"{name}: SHA-256 mismatch ({digest} != {spec['sha256']})")

    files = safe_extract(archive, out_dir)
    # Some archives nest further zips (PhiUSIIL, Mendeley); extract one level deeper.
    for inner in list(out_dir.rglob("*.zip")):
        if inner != archive:
            files += safe_extract(inner, inner.parent)

    return {
        "url": spec["url"],
        "licence": spec["licence"],
        "citation": spec["citation"],
        "sha256": digest,
        "files": sorted(files),
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", default="data/raw", type=Path)
    ap.add_argument("--only", nargs="*", choices=list(SOURCES), help="subset of sources")
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    args = ap.parse_args()

    manifest_path = args.raw_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name in args.only or SOURCES:
        manifest[name] = download(name, args.raw_dir, args.force)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[download] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
