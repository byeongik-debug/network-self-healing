"""Repeat one non-2026 run and compare parameters, metrics, and trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
import torch

from experiments.evaluate_dqn import EvaluationConfig
from experiments.evaluate_shielded_dqn import ShieldedDQNEvaluator
from experiments.train_dqn import TrainingConfig
from experiments.train_safe_dqn_stepbudget import StepBudgetSafeDQNTrainer


def main():
    root = Path("results/multiseed")
    repeat_dir = root / "repro_seed_2027"
    config = TrainingConfig(episodes=1, seed=2027, split_seed=2026,
                            output_dir=str(repeat_dir))
    started = perf_counter()
    repeated = StepBudgetSafeDQNTrainer(config, 35747).train()
    original_path = root / "safe_seed_2027.pt"
    original = torch.load(original_path, map_location="cpu", weights_only=False)
    rerun = torch.load(repeated.checkpoint_path, map_location="cpu", weights_only=False)
    parameter_equal = all(torch.equal(original["online_network_state_dict"][key], value)
                          for key,value in rerun["online_network_state_dict"].items())
    target_equal = all(torch.equal(original["target_network_state_dict"][key], value)
                       for key,value in rerun["target_network_state_dict"].items())
    first = ShieldedDQNEvaluator(EvaluationConfig(checkpoint_path=str(original_path))).evaluate(save=False)
    second = ShieldedDQNEvaluator(EvaluationConfig(checkpoint_path=str(repeated.checkpoint_path))).evaluate(save=False)
    rows_equal = first["rows"] == second["rows"]
    trajectories_equal = first["trajectories"] == second["trajectories"]
    result = {"algorithm":"safe","seed":2027,"global_step":repeated.global_step,
        "parameters_bitwise_equal":parameter_equal,"target_parameters_bitwise_equal":target_equal,
        "evaluation_rows_equal":rows_equal,"trajectories_equal":trajectories_equal,
        "repeat_training_runtime_seconds":perf_counter()-started,
        "repeat_checkpoint":str(repeated.checkpoint_path)}
    (root/"multiseed_reproducibility.json").write_text(
        json.dumps(result,indent=2),encoding="utf-8")
    summary_path=root/"multiseed_statistical_summary.json"
    summary=json.loads(summary_path.read_text(encoding="utf-8"))
    summary["reproducibility"]=result
    summary_path.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__": main()
