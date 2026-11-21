import sys
from types import ModuleType

import numpy as np
import pytest

from modules.optimizers.QuantumDirectionSampler import QuantumDirectionSampler


class FakeCircuit:
    def __init__(self, count):
        assert count == 1
    def h(self, index):
        assert index == 0


class FakeState:
    def __init__(self):
        self.seed_value = None
    @classmethod
    def from_instruction(cls, circuit):
        assert isinstance(circuit, FakeCircuit)
        return cls()
    def seed(self, seed):
        self.seed_value = seed
    def sample_memory(self, shots):
        return ["0", "1"] * (shots // 2) + (["0"] if shots % 2 else [])


def test_quantum_direction_sampler_converts_bits_and_seeds(monkeypatch):
    qiskit = ModuleType("qiskit")
    qiskit.QuantumCircuit = FakeCircuit
    quantum_info = ModuleType("qiskit.quantum_info")
    quantum_info.Statevector = FakeState
    monkeypatch.setitem(sys.modules, "qiskit", qiskit)
    monkeypatch.setitem(sys.modules, "qiskit.quantum_info", quantum_info)
    sampler = QuantumDirectionSampler()
    values = sampler.sample(3, seed=42)
    np.testing.assert_array_equal(values, np.array([-1, 1, -1], dtype=np.int8))
    assert sampler.state.seed_value == 42
    with pytest.raises(ValueError, match="positive"):
        sampler.sample(0, seed=42)
