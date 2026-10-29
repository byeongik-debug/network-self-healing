import math
from pathlib import Path

from experiments.train_graph_dqn import GraphDQNTrainer, GraphTrainingConfig, _schedule


def test_schedule_hits_endpoints():
    assert _schedule(0.8, 0.1, 0, 100) == 0.8
    assert _schedule(0.8, 0.1, 100, 100) == 0.1
    assert _schedule(0.8, 0.1, 200, 100) == 0.1


def test_graph_trainer_smoke_runs_updates_without_damage():
    config = GraphTrainingConfig(
        episodes=4, batch_size=2, replay_warmup=2, max_steps=8,
        hidden=16, guidance_start=1.0, guidance_end=1.0,
        save_outputs=False, seed=17,
    )
    trainer = GraphDQNTrainer(config)
    logs = trainer.train()
    assert len(logs) == 4
    assert sum(log.optimizer_updates for log in logs) > 0
    assert sum(log.new_policy_damage_events for log in logs) == 0
    assert all(math.isfinite(log.episode_return) for log in logs)


def test_no_guidance_run_has_zero_guided_actions_and_distinct_name(tmp_path: Path):
    config = GraphTrainingConfig(
        episodes=2, batch_size=2, replay_warmup=2, max_steps=4, hidden=16,
        guidance_start=0.0, guidance_end=0.0, output_dir=str(tmp_path), seed=19,
    )
    logs = GraphDQNTrainer(config).train()
    assert sum(log.guided_actions for log in logs) == 0
    assert (tmp_path / "model_seed_19_ep2_noguidance.pt").exists()
    assert (tmp_path / "training_seed_19_ep2_noguidance.csv").exists()
