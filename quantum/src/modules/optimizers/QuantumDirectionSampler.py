from __future__ import annotations

import numpy as np

import modules.common.TypeContracts as tc


class QuantumDirectionSampler:
    @tc.validate_types
    def __init__(self) -> None:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector

        circuit = QuantumCircuit(1)
        circuit.h(0)
        self.state = Statevector.from_instruction(circuit)

    @tc.validate_types
    def sample(self, count: int, seed: int) -> np.ndarray:

        if count < 1:
            raise ValueError("count must be positive")

        self.state.seed(int(seed))
        memory = self.state.sample_memory(shots=int(count))

        return np.fromiter(
            (1 if bit == "1" else -1 for bit in memory),
            dtype=np.int8,
            count=count,
        )
