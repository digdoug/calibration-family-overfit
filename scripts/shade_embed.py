"""SHADE step 2: embed exported SHADE transcripts -> track2-schema npz.

Reads results/shade_export/{benign,attack}/{family}.jsonl (from shade_export.py),
keeps only the pre-registered COMMON task set (>=3 benign AND >=3 attack
unfiltered in every family; the election-tampering pair excluded upstream,
following the public SHADE-Arena paper's own exclusion), embeds with
OpenAI text-embedding-3-small via track2_embed.embed() (token-aware batching +
429 retry, 8k-token truncation), and writes per-family npz in the track2 schema
{emb, label, task_id} plus filter masks {main_ok, keep_side, keep_strict} so the
filtered sensitivity cuts can be materialized without re-embedding.

DATA HANDLING: inputs/outputs live under gitignored results/ only. The embedder
is OpenAI (no-train on API data) -- permitted under the aggregate-only data agreement (no-train API only).

Run: .venv-controlarena/bin/python scripts/shade_embed.py [--action-only] [--hint]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import track2_embed as t2  # reuse embed() + truncation conventions
from openai import OpenAI

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTION_ONLY = "--action-only" in sys.argv
USE_HINT = "--hint" in sys.argv
EXPORT = os.path.join(REPO, "results", "shade_export" + ("_hint" if USE_HINT else ""))
OUT = os.path.join(REPO, "results", "track2_shade"
                   + ("_hint" if USE_HINT else "")
                   + ("_actiononly" if ACTION_ONLY else ""))
FAMILIES = ["shade_claude35s", "shade_gpt41", "shade_gemini25f"]
MIN_PER_CLASS = 3


def load(fam: str, mode: str) -> list[dict]:
    with open(os.path.join(EXPORT, mode, f"{fam}.jsonl")) as f:
        return [json.loads(l) for l in f]


def main() -> int:
    recs = {fam: load(fam, "benign") + load(fam, "attack") for fam in FAMILIES}

    # pre-registered common-task rule
    common = None
    for fam, rs in recs.items():
        per = {}
        for r in rs:
            k = per.setdefault(r["task_id"], [0, 0])
            k[r["label"]] += 1
        ok = {t for t, (b, a) in per.items()
              if b >= MIN_PER_CLASS and a >= MIN_PER_CLASS}
        common = ok if common is None else common & ok
    common = sorted(common)
    print(f"common tasks: {len(common)}")

    os.makedirs(OUT, exist_ok=True)
    client = OpenAI()
    for fam in FAMILIES:
        rs = [r for r in recs[fam] if r["task_id"] in common]
        texts = [t2._truncate(r["action_text" if ACTION_ONLY else "text"])
                 or "(empty)" for r in rs]
        print(f"=== {fam}: {len(rs)} records "
              f"({sum(r['label'] for r in rs)} attack) ===", flush=True)
        emb = t2.embed(client, texts)
        np.savez(os.path.join(OUT, f"{fam}.npz"),
                 emb=emb,
                 label=np.array([r["label"] for r in rs]),
                 task_id=np.array([r["task_id"] for r in rs], dtype=object),
                 main_ok=np.array([bool(r["main_ok"]) for r in rs]),
                 keep_side=np.array([bool(r["keep_side"]) for r in rs]),
                 keep_strict=np.array([bool(r["keep_strict"]) for r in rs]))
        print(f"  saved {fam}.npz emb={emb.shape}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
