from collections import Counter
import pytest
from qrl_navigation.replay_buffer import ReplayWeightedBuffer


@pytest.mark.slow
def test_sampling():
    w_buffer = ReplayWeightedBuffer(buffer_size=10, batch_size=5)
    for value in [10, 20, 30, 40, 50]:
        w_buffer.add(value, value, value, value, value, err=value)

    all_states = []
    for iteration in range(100):
        states, actions, rewards, next_states, dones, probs, idxs = w_buffer.sample()
        states = states.squeeze().numpy().tolist()
        all_states += states

    counts = Counter(all_states)
    assert list(map(lambda item: item[0], counts.most_common())) == [50.0, 40.0, 30.0, 20.0, 10.0]


def test_check_ready_for_samplig():
    w_buffer = ReplayWeightedBuffer(buffer_size=10, batch_size=5)
    for value in range(4):
        w_buffer.add(value, value, value, value, value, err=value)
    assert w_buffer.is_ready_to_sample() is False

    for value in range(4):
        w_buffer.add(value, value, value, value, value, err=value)
    assert w_buffer.is_ready_to_sample() is True


def test_correct_update():
    w_buffer = ReplayWeightedBuffer(buffer_size=5, batch_size=5)
    for value in range(10):
        w_buffer.add(value, value, value, value, value, err=value)
    assert w_buffer.memory[0].state == 5
    assert w_buffer.errs[0] == 5.01

    w_buffer.update_errs(values=[0, -1], indexes=[0, 3])
    assert w_buffer.errs[0] == 0.01
    assert w_buffer.errs[3] == 1.01


def test_fillin():
    w_buffer = ReplayWeightedBuffer(buffer_size=2, batch_size=2)
    for value in [10, 20, 30, 40]:
        w_buffer.add(value, value, value, value, value, err=value)
    assert w_buffer.memory[0].state == 30
    assert w_buffer.memory[1].state == 40
