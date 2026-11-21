"""Unit tests for AdaptiveQuantumDirectionSampler."""
import numpy as np
import pytest
from modules.optimizers.AdaptiveQuantumDirectionSampler import AdaptiveQuantumDirectionSampler


def test_update_accepts_signed_directions_and_reports_probability():
    sampler = AdaptiveQuantumDirectionSampler(learning_rate=0.1)
    result = sampler.update(np.array([-1, 1, 1, -1], dtype=np.int8), 0.5)
    assert result["reward"] == 0.5
    assert 0.0 < sampler.probability_plus < 1.0
    assert np.isfinite(sampler.theta)


def test_update_rejects_normalized_directions():
    sampler = AdaptiveQuantumDirectionSampler()
    with pytest.raises(ValueError, match="directions"):
        sampler.update(np.array([-0.5, 0.5]), 0.1)


def test_update_rejects_nonfinite_reward():
    sampler = AdaptiveQuantumDirectionSampler()
    with pytest.raises(ValueError, match="finite"):
        sampler.update(np.array([-1, 1], dtype=np.int8), float("nan"))


def test_sampler_learns_independent_parameter_probabilities():
    """Opposite rewards for distinct coordinates should not share one theta."""
    sampler = AdaptiveQuantumDirectionSampler(learning_rate=0.1)
    sampler.prepare(2)
    sampler.update(np.array([1, -1], dtype=np.int8), 0.5)
    probabilities = np.asarray(sampler.probabilities_plus)
    assert probabilities.shape == (2,)
    assert probabilities[0] > 0.5
    assert probabilities[1] < 0.5


def test_grouped_theta_sizes_and_probabilities():
    for group_size, expected in [(1, 10), (3, 4), (20, 1), (None, 1)]:
        sampler = AdaptiveQuantumDirectionSampler(theta_group_size=group_size)
        sampler.prepare(10)
        assert len(sampler.theta) == expected
        assert len(sampler.probabilities_plus) == expected


def test_grouped_update_uses_group_means():
    sampler = AdaptiveQuantumDirectionSampler(theta_group_size=2, learning_rate=0.1)
    sampler.prepare(4)
    sampler.update(np.array([1, 1, -1, -1], dtype=np.int8), 0.5)
    assert sampler.probabilities_plus[0] > 0.5
    assert sampler.probabilities_plus[1] < 0.5


def test_grouped_sampler_rejects_invalid_sizes():
    for group_size in (0, -1, 1.5):
        with pytest.raises(ValueError, match="theta_group_size"):
            AdaptiveQuantumDirectionSampler(theta_group_size=group_size)


def test_grouped_sampler_rejects_changed_parameter_count():
    sampler = AdaptiveQuantumDirectionSampler(theta_group_size=4)
    sampler.prepare(8)
    with pytest.raises(ValueError, match="parameter count changed"):
        sampler.prepare(7)
