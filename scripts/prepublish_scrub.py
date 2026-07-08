#!/usr/bin/env python3
"""Pre-publish release gate for the public artifact.

Design rule: before the repo (or a curated
snapshot of it) is ever made public, it must contain NO ControlArena/SHADE
canary GUID, NO raw `.eval`/`.jsonl`/`.npz` sabotage transcript, and no SHADE
task text.

Design note on canaries: the whole point of a canary string is that it must NOT
appear in any public/trainable artifact, so this checker cannot store the literal
GUIDs (that would re-pollute the tree it guards). Instead it stores the SHA-256
of each known canary and hashes every GUID/token-shaped candidate it finds,
flagging matches. The authoritative literal list lives only in the upstream
ControlArena repo README.

Usage:
    python3 scripts/prepublish_scrub.py            # audit the git-tracked tree
    python3 scripts/prepublish_scrub.py PATH ...   # audit specific paths/dirs

Exit code 0 = clean (publishable); 1 = at least one violation found.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

# SHA-256 of the three ControlArena canary strings (MAPS-evals / benchmark-data /
# ControlArena). Literal strings deliberately absent — see module docstring.
CANARY_HASHES = {
    "9c3b98584803c7c61893807ef7c046194d38b605a58fe48ab859c8dcb9b04380",
    "2c7c9d047919c8e41df91d6f40f34afc8c8d6661d369a0676bcb5edad38b6452",
    "530c5ab6e540b1602e5ca4cdc18e50b154f389fba4fc24316f0ee77f3c34632b",
}

# Token shapes a canary GUID can take: a UUID, or a long alphanumeric blob.
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{15,}")

# Raw-transcript / eval artifacts that must never enter the public set.
RAW_DATA_SUFFIXES = {".eval", ".npz"}
# .jsonl is raw transcripts ONLY under data/ or results/; allow elsewhere.
RAW_JSONL_DIRS = ("data/", "results/", "trajectories/")

# Files this checker may legitimately match its own hashes/placeholders in.
SELF_ALLOW = {"scripts/prepublish_scrub.py"}


def tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout
    return [Path(p) for p in out.splitlines() if p]


def iter_targets(args: list[str]) -> list[Path]:
    if not args:
        return tracked_files()
    files: list[Path] = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            files.extend(q for q in p.rglob("*") if q.is_file())
        elif p.is_file():
            files.append(p)
    return files


def scan_canary(path: Path, violations: list[str]) -> None:
    try:
        text = path.read_text(errors="ignore")
    except (OSError, UnicodeError):
        return
    for tok in TOKEN_RE.findall(text):
        h = hashlib.sha256(tok.encode()).hexdigest()
        if h in CANARY_HASHES:
            violations.append(f"CANARY GUID reproduced verbatim in {path}")
            return


def scan_raw_data(path: Path, violations: list[str]) -> None:
    posix = path.as_posix()
    if path.suffix in RAW_DATA_SUFFIXES:
        violations.append(f"RAW artifact ({path.suffix}) present: {path}")
    elif path.suffix == ".jsonl" and any(posix.startswith(d) for d in RAW_JSONL_DIRS):
        violations.append(f"RAW transcript (.jsonl under a data dir): {path}")


def main(argv: list[str]) -> int:
    targets = iter_targets(argv)
    violations: list[str] = []
    for path in targets:
        if path.as_posix() in SELF_ALLOW:
            continue
        scan_raw_data(path, violations)
        # Only text-scan reasonably sized files for canaries.
        try:
            if path.stat().st_size <= 5_000_000:
                scan_canary(path, violations)
        except OSError:
            pass

    print(f"prepublish_scrub: audited {len(targets)} files")
    if violations:
        print(f"\nFAIL — {len(violations)} violation(s):")
        for v in violations:
            print(f"  - {v}")
        print("\nDo NOT publish until these are resolved.")
        return 1
    print("PASS — no canary GUID, raw transcript, or .eval found. Publishable.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
