"""Check local model and config files against the frozen fingerprints.

    python -m fraudshield.verify      (or: make verify-models)

Compares SHA-256 hashes with reports/metrics/frozen_models.json, written when the
models were frozen for the one-time test evaluation. A match means these are the
exact files the reported test results came from.
"""

from __future__ import annotations

import hashlib
import json
import sys

from fraudshield.utils import resolve


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    frozen = json.loads(resolve("reports/metrics/frozen_models.json").read_text())
    ok = True
    for rel, expected in frozen["sha256"].items():
        path = resolve(rel)
        if not path.exists():
            print(f"MISSING   {rel}")
            ok = False
        elif sha256(path) != expected:
            print(f"DIFFERENT {rel}")
            ok = False
        else:
            print(f"ok        {rel}")
    print("All files match the frozen models." if ok else
          "Some files are missing or different: these are not the frozen models.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
