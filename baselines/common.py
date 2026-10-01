"""Shared baseline interfaces and execution results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class RecoveryEnvironment(Protocol):
    """Only the public surface available to recovery algorithms."""

    max_steps: int

    def get_observation(self) -> dict[str, Any]: ...
    def get_valid_actions(self) -> list[dict[str, Any]]: ...
    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]: ...


@dataclass(frozen=True)
class RecoveryResult:
    algorithm: str
    success: bool
    terminated: bool
    truncated: bool
    steps: int


class ActionExecutor:
    """Runs public discrete actions while respecting the episode limit."""

    def __init__(self, env: RecoveryEnvironment, max_steps: int) -> None:
        self.env = env
        self.max_steps = min(max_steps, env.max_steps)
        self.steps = 0
        self.terminated = False
        self.truncated = False
        self.success = False

    def find(self, kind: str, **fields: object) -> int | None:
        for action in self.env.get_valid_actions():
            if action["kind"] == kind and all(action.get(key) == value for key, value in fields.items()):
                return int(action["id"])
        return None

    def run(self, action_id: int | None) -> dict[str, Any] | None:
        if action_id is None or self.done or self.steps >= self.max_steps:
            return None
        _, _, self.terminated, self.truncated, info = self.env.step(action_id)
        self.steps += 1
        if self.terminated:
            self.success = bool(info.get("recovery_success", False))
        return info

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated or self.steps >= self.max_steps

    def result(self, algorithm: str) -> RecoveryResult:
        return RecoveryResult(algorithm, self.success, self.terminated, self.truncated, self.steps)

