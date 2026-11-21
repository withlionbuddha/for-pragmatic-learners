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
