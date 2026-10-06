from experiments.train_dqn import TrainingConfig
from experiments.train_vanilla_dqn_stepbudget import StepBudgetVanillaDQNTrainer


def test_vanilla_exact_budget_and_seed_reproducibility(tmp_path):
    def train(directory):
        config = TrainingConfig(episodes=1, batch_size=2, replay_capacity=100,
            replay_warmup=2, epsilon_decay_steps=20, target_sync_interval=5,
            max_episode_steps=4, seed=2035, output_dir=str(directory),
            save_csv=False, save_checkpoint=False)
        return StepBudgetVanillaDQNTrainer(config, 17).train()
    first, second = train(tmp_path / "a"), train(tmp_path / "b")
    assert first["global_step"] == second["global_step"] == 17
    assert first["sampled_scenario_ids"] == second["sampled_scenario_ids"]
    assert first["logs"] == second["logs"]
    assert first["logs"][-1].budget_cut
    assert not first["logs"][-1].completed
