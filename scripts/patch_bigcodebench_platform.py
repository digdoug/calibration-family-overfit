"""Idempotent patch: force the bigcodebench compose service to linux/amd64.

The `bigcodebench/bigcodebench-evaluate:latest` image is amd64-ONLY (no arm64 manifest). On
Apple Silicon, `docker compose up` fails with "no matching manifest for linux/arm64/v8" and EVERY
sample errors (RuntimeError: No services started). Pinning the service platform to linux/amd64
makes Docker pull + run the amd64 image under emulation (Rosetta/QEMU) — slower, but functional
for the short backdoor-verification execs.

vendor/ is gitignored, so this lives as a re-appliable patch (like scripts/patch_bash_scorer.py).
Re-apply after any `pip install -e vendor/control-arena` reinstall. Disclose the emulation in the
paper's per-setting generation notes.

Run: .venv-controlarena/bin/python scripts/patch_bigcodebench_platform.py
"""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMPOSE = REPO / "vendor/control-arena/control_arena/settings/bigcodebench/compose.yml"


def main():
    text = COMPOSE.read_text()
    if "platform: linux/amd64" in text:
        print("already patched (platform: linux/amd64 present)")
        return
    out = []
    patched = False
    for ln in text.splitlines():
        out.append(ln)
        if (not patched) and ln.strip().startswith("image:") and "bigcodebench" in ln:
            indent = ln[: len(ln) - len(ln.lstrip())]
            out.append(f"{indent}platform: linux/amd64  # amd64-only image; emulate on Apple Silicon (patch_bigcodebench_platform.py)")
            patched = True
    if not patched:
        raise SystemExit("ERROR: could not find the bigcodebench image: line to patch")
    COMPOSE.write_text("\n".join(out) + "\n")
    print(f"patched: added 'platform: linux/amd64' to {COMPOSE}")


if __name__ == "__main__":
    raise SystemExit(main())
