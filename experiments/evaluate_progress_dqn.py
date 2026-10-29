"""Evaluate a frozen DQN checkpoint under the progress viability shield."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.evaluate_dqn import EvaluationConfig
from experiments.evaluate_shielded_dqn import ShieldedDQNEvaluator
from safety.progress_viability_shield import ProgressViabilityShield


def evaluate(checkpoint: str, output_dir: str) -> dict[str, object]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    result = ShieldedDQNEvaluator(
        EvaluationConfig(checkpoint_path=checkpoint),
        shield=ProgressViabilityShield(),
        output_csv=str(root / "evaluation.csv"),
        output_summary=str(root / "summary.json"),
        output_trajectories=str(root / "trajectories.json"),
    ).evaluate(save=True)
    return result["summary"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    summary = evaluate(args.checkpoint, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
