"""
W1 GATE 1 smoke test (the design doc s.7).

Minimal end-to-end exercise of the ControlArena -> Inspect -> trajectory-
export pipeline:

  1. Run a single APPS task in HONEST and ATTACK modes under the
     defer-to-trusted micro-protocol, with gpt-4.1-mini for both the
     trusted and untrusted policy. Docker sandbox.
  2. Confirm an eval log file is produced.
  3. (Done by the wrapper) `control-arena trajectory export` round-trips
     the log into trajectory.jsonl + tools.json + metadata.json.

This is NOT the GATE 1 hard check (single-family baseline AUROC matching
a published figure). This is the prerequisite plumbing check. Cost: <$1.

Run from the project root with the ControlArena venv:
    .venv-controlarena/bin/python scripts/w1_smoke.py
"""

from __future__ import annotations

import os
from pathlib import Path

from inspect_ai import eval as inspect_eval

from control_arena import EvalMode
from control_arena.eval._control_eval_config import ControlEvalConfig
from control_arena.eval._control_eval_def import get_control_task_product
from control_arena.micro_protocol import defer_to_trusted_builder
from control_arena.settings.apps.apps_setting import AppsSetting


LOG_DIR = Path(__file__).resolve().parents[1] / "results" / "W1_smoke"


def main() -> int:
    assert os.environ.get("OPENAI_API_KEY"), (
        "OPENAI_API_KEY not set; source ~/.zshrc before running."
    )

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    eval_config = ControlEvalConfig(
        trusted_model="openai/gpt-4.1-mini",
        untrusted_model="openai/gpt-4.1-mini",
        limit=1,
        max_sandboxes=2,
        max_tasks=2,
        log_dir=str(LOG_DIR),
    )

    setting = AppsSetting(sandbox_type="docker")
    protocol = defer_to_trusted_builder()

    eval_tasks = get_control_task_product(
        settings=setting,
        micro_protocols=protocol,
        modes=[EvalMode.HONEST, EvalMode.ATTACK],
        eval_config=eval_config,
    )

    print(f"Running {len(eval_tasks)} task(s); logs -> {LOG_DIR}")
    logs = inspect_eval(tasks=eval_tasks, log_dir=str(LOG_DIR), limit=1)

    print("\n--- eval results ---")
    for i, log in enumerate(logs):
        print(f"  log[{i}]: location={getattr(log, 'location', '?')}, status={getattr(log, 'status', '?')}")

    print(f"\nNext: control-arena trajectory export {LOG_DIR} {LOG_DIR.parent / 'W1_smoke_exported'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
