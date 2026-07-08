"""Idempotent patch: make ControlArena's honest+attack policies honor a
CA_FORCE_TOOL_CHOICE env var on their model.generate() call.

WHY: some open-weight policies (Gemma-3) emit unparsed ```tool_code text on
tool_choice=auto and never submit; forcing tool_choice="any" (=OpenAI 'required')
makes them emit clean structured tool calls (verified: valid code on DeepInfra).
The policy builders don't expose tool_choice, so we inject it here.

SAFE: env unset -> None -> identical to current default (auto). Only activates
when CA_FORCE_TOOL_CHOICE=any is set (via w2_generate_family.py --untrusted-force-tool-choice).
Applied to the EDITABLE vendored install; re-run after any reinstall. Idempotent.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
POL = REPO / "vendor" / "control-arena" / "control_arena" / "policy"
INJECT = '(__import__("os").environ.get("CA_FORCE_TOOL_CHOICE") or None)'
MARKER = "CA_FORCE_TOOL_CHOICE"

HONEST = POL / "_honest_policy.py"
ATTACK = POL / "_attack_policy.py"

# honest: single-line generate call
H_OLD = ".generate(state._messages, tools=tools, cache=cache)"
H_NEW = f".generate(state._messages, tools=tools, tool_choice={INJECT}, cache=cache)"

# attack: multi-line generate call
A_OLD = "            input=messages,\n            tools=tools,\n            cache=cache,"
A_NEW = f"            input=messages,\n            tools=tools,\n            tool_choice={INJECT},\n            cache=cache,"


def patch(path: Path, old: str, new: str) -> str:
    text = path.read_text()
    if MARKER in text:
        return f"  {path.name}: already patched"
    if old not in text:
        return f"  {path.name}: ANCHOR NOT FOUND (policy source changed?) -> manual check"
    path.write_text(text.replace(old, new, 1))
    return f"  {path.name}: PATCHED"


if __name__ == "__main__":
    print("Patching policies to honor CA_FORCE_TOOL_CHOICE:")
    print(patch(HONEST, H_OLD, H_NEW))
    print(patch(ATTACK, A_OLD, A_NEW))
    # verify
    for p in (HONEST, ATTACK):
        ok = MARKER in p.read_text()
        print(f"  verify {p.name}: {'OK' if ok else 'MISSING'}")
