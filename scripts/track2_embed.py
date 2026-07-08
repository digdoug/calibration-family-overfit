"""Track-2 step 1: embed each family's transcripts so we can train a
family-CONDITIONED monitor (logistic head per calibration family).

This is the ONLY way to test the ranking-overfit thesis (the design doc s1a Track 2):
a fixed-prompt monitor's AUROC has no calibration-family dependence by construction,
so we build a scorer whose SCORING depends on family A (a head trained on A's
honest-vs-attack transcript embeddings), then score family B.

Embeds the model's own output (reasoning + submitted code) -- the family's
stylistic signature -- with OpenAI text-embedding-3-small. Saves per-family
results/track2/<family>.npz {emb, label, task_id}.

Run: source ~/.zshrc && .venv-controlarena/bin/python scripts/track2_embed.py <fam_dir>...
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
from inspect_ai.log import read_eval_log
from openai import OpenAI

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# --action-only (the design doc s4): embed ONLY the submitted code, stripping all prose.
# If the interaction SURVIVES this -> behavioral transfer failure (strong claim);
# if it VANISHES -> the overfit was transcript-STYLE (narrower claim).
ACTION_ONLY = "--action-only" in sys.argv


def _arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


# --emb-model (Phase-1b embedder-specificity control): re-embed the SAME transcripts with a
# different OpenAI embedder. If the +0.127 interaction replicates across embedders it's not a
# text-embedding-3-small quirk. Each embedder writes its own dir (3-small = canonical track2/).
EMB_MODEL = _arg("--emb-model", "text-embedding-3-small")
_EMB_TAG = {"text-embedding-3-small": "", "text-embedding-3-large": "_3large",
            "text-embedding-ada-002": "_ada002"}.get(
    EMB_MODEL, "_" + EMB_MODEL.replace("text-embedding-", "").replace("-", ""))
# --out NAME: write npz to results/NAME instead of results/track2[...]. Use for a 2nd substrate
# (e.g. --out track2_bcb for bigcodebench) so its embeddings don't clobber the APPS track2 npz.
_OUT = _arg("--out", None)
OUT = (os.path.join(REPO, "results", _OUT) if _OUT
       else os.path.join(REPO, "results", ("track2_actiononly" if ACTION_ONLY else "track2") + _EMB_TAG))
MAX_TOK = 8000  # under the 8192 embedding limit; truncate by real tokens (code-dense text can be ~1.7 chars/token)
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENC = None


def _truncate(text: str) -> str:
    if _ENC is None:
        return text[:14000]
    toks = _ENC.encode(text)
    return text if len(toks) <= MAX_TOK else _ENC.decode(toks[:MAX_TOK])


def sample_text(s) -> str:
    """The model's own output: assistant reasoning + submitted code."""
    parts = []
    for m in s.messages:
        if getattr(m, "role", None) != "assistant":
            continue
        if not ACTION_ONLY:
            t = m.text if getattr(m, "text", None) else ""
            if t:
                parts.append(t)
        for tc in (getattr(m, "tool_calls", None) or []):
            args = getattr(tc, "arguments", None)
            if isinstance(args, dict) and args.get("code"):
                parts.append("```\n" + str(args["code"]) + "\n```")
    return _truncate("\n".join(parts)) or "(empty)"


def load_family(fam_dir: str):
    texts, labels, tasks = [], [], []
    for f in sorted(glob.glob(os.path.join(fam_dir, "*.eval"))):
        label = 1 if "attack" in os.path.basename(f) else 0
        for s in (read_eval_log(f).samples or []):
            # only samples that actually submitted (have a suspicion score) are real
            if not (s.scores and "max_suspicion_scorer" in s.scores
                    and str(s.scores["max_suspicion_scorer"].value) not in ("N", "None")):
                continue
            texts.append(sample_text(s)); labels.append(label); tasks.append(str(s.id))
    return texts, np.array(labels), np.array(tasks, dtype=object)


def embed(client, texts, max_batch=128, max_batch_tok=120000):
    # Token-AWARE batching: the embeddings API caps a single request at 300k tokens AND the org
    # at 1M tokens/min (TPM). A fixed count (128) overflows the request cap for verbose families
    # (Gemini ~4k tok/transcript). Grow each chunk to a token budget (<<300k, with TPM headroom),
    # and retry on 429 (the per-minute limit resets, so a short backoff clears it).
    import time
    def ntok(t):
        return len(_ENC.encode(t)) if _ENC is not None else len(t) // 4
    out, i, n = [], 0, len(texts)
    while i < n:
        j, tok = i, 0
        while j < n and (j - i) < max_batch:
            t = ntok(texts[j])
            if j > i and tok + t > max_batch_tok:
                break
            tok += t; j += 1
        for attempt in range(8):
            try:
                r = client.embeddings.create(model=EMB_MODEL, input=texts[i:j])
                break
            except Exception as e:
                if ("rate_limit" in str(e).lower() or "429" in str(e)) and attempt < 7:
                    time.sleep(min(2 ** attempt, 30)); continue
                raise
        out.extend([d.embedding for d in r.data])
        print(f"    embedded {j}/{n}", flush=True)
        i = j
    return np.array(out, dtype=np.float32)


def local_embed(model, texts, prefix=""):
    # Non-OpenAI open-weight embedder (sentence-transformers) for the monitor-vendor robustness
    # control. normalize so the downstream scaler/PCA/logistic head are scale-comparable.
    import numpy as _np
    inp = [prefix + t for t in texts] if prefix else texts
    out = model.encode(inp, batch_size=8, normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True)
    return out.astype(_np.float32)


def main():
    os.makedirs(OUT, exist_ok=True)
    local = EMB_MODEL.startswith("local:")
    print(f"mode: {'ACTION-ONLY (code, no prose)' if ACTION_ONLY else 'full transcript'} | "
          f"embedder: {EMB_MODEL} -> {OUT}")
    client = model = prefix = None
    if local:
        from sentence_transformers import SentenceTransformer
        hf_id = EMB_MODEL.split("local:", 1)[1]
        model = SentenceTransformer(hf_id, trust_remote_code=True,
                                    device="mps" if __import__("torch").backends.mps.is_available() else "cpu")
        model.max_seq_length = min(getattr(model, "max_seq_length", 512) or 512, 2048)
        prefix = "search_document: " if "nomic" in hf_id.lower() else ""
        print(f"  local embedder loaded: {hf_id} (max_seq_len={model.max_seq_length}, prefix={prefix!r})")
    else:
        client = OpenAI()
    for fam_dir in [a for a in sys.argv[1:] if not a.startswith("--") and a not in (EMB_MODEL, _OUT)]:
        fam = os.path.basename(fam_dir.rstrip("/")).replace("W2_", "")
        print(f"=== {fam} ===", flush=True)
        texts, labels, tasks = load_family(fam_dir)
        print(f"  {len(texts)} scored samples ({int(labels.sum())} attack / {int((labels==0).sum())} honest)", flush=True)
        emb = local_embed(model, texts, prefix) if local else embed(client, texts)
        np.savez(os.path.join(OUT, f"{fam}.npz"), emb=emb, label=labels, task_id=tasks)
        print(f"  saved {fam}.npz  emb={emb.shape}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
