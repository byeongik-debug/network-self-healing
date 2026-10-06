from __future__ import annotations

import numpy as np
import pytest

from agents.replay_buffer import ReplayBuffer


def transition(index: int = 0) -> tuple[object, ...]:
    state = np.full(79, index, dtype=np.float32)
    next_state = np.full(79, index + 1, dtype=np.float32)
    mask = np.ones(19, dtype=bool)
    mask[index % 19] = False
    return state, index % 19, float(index), next_state, index % 2 == 0, index % 3 == 0, mask


def test_push_and_len_preserve_separate_end_flags() -> None:
    buffer = ReplayBuffer(capacity=3)
    buffer.push(*transition(1))
    assert len(buffer) == 1

    states, actions, rewards, next_states, terminated, truncated, masks = buffer.sample(1)
    assert states[0, 0] == 1
    assert actions.tolist() == [1]
    assert rewards.tolist() == [1.0]
    assert next_states[0, 0] == 2
    assert terminated.tolist() == [False]
    assert truncated.tolist() == [False]
    assert masks[0].sum() == 18


def test_capacity_removes_oldest_transition() -> None:
    buffer = ReplayBuffer(capacity=2)
    for index in range(3):
        buffer.push(*transition(index))

    states, *_ = buffer.sample(2)
    assert len(buffer) == 2
    assert set(states[:, 0].tolist()) == {1.0, 2.0}


def test_sample_shapes_and_dtypes() -> None:
    buffer = ReplayBuffer(capacity=64)
    for index in range(64):
        buffer.push(*transition(index))

    states, actions, rewards, next_states, terminated, truncated, masks = buffer.sample(64)
    assert states.shape == (64, 79) and states.dtype == np.float32
    assert actions.shape == (64,) and actions.dtype == np.int64
    assert rewards.shape == (64,) and rewards.dtype == np.float32
    assert next_states.shape == (64, 79) and next_states.dtype == np.float32
    assert terminated.shape == (64,) and terminated.dtype == np.bool_
    assert truncated.shape == (64,) and truncated.dtype == np.bool_
    assert masks.shape == (64, 19) and masks.dtype == np.bool_


def test_push_copies_state_next_state_and_action_mask() -> None:
    buffer = ReplayBuffer(capacity=1)
    values = list(transition(4))
    state, next_state, mask = values[0], values[3], values[6]
    buffer.push(*values)

    state[:] = -1
    next_state[:] = -1
    mask[:] = False
    states, _, _, next_states, _, _, masks = buffer.sample(1)
    assert np.all(states == 4)
    assert np.all(next_states == 5)
    assert masks[0].sum() == 18


def test_terminated_and_truncated_are_stored_independently() -> None:
    buffer = ReplayBuffer(capacity=2)
    first = list(transition(1))
    first[4], first[5] = True, False
    second = list(transition(2))
    second[4], second[5] = False, True
    buffer.push(*first)
    buffer.push(*second)

    states, _, _, _, terminated, truncated, _ = buffer.sample(2)
    flags_by_state = {
        int(state[0]): (bool(term), bool(trunc))
        for state, term, trunc in zip(states, terminated, truncated)
    }
    assert flags_by_state == {1: (True, False), 2: (False, True)}


def test_sampling_more_than_buffer_size_raises() -> None:
    buffer = ReplayBuffer(capacity=2)
    buffer.push(*transition())
    with pytest.raises(ValueError, match="cannot sample 2 transitions"):
        buffer.sample(2)


@pytest.mark.parametrize("capacity", [0, -1])
def test_capacity_must_be_positive(capacity: int) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        ReplayBuffer(capacity)
