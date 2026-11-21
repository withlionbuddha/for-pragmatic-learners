"""Contract tests for the quantum optimizer adapter."""

from modules.adapters.QuantumAdapter import QuantumAdapter


class FakeOptimizer:
    def __init__(self):
        self.model = None
        self.evaluate = None

    def initialize_parameters(self, model):
        self.model = model

    def run_objective(self, evaluate):
        self.evaluate = evaluate
        return {"status": "completed", "loss": evaluate()}


def test_adapter_translates_training_calls():
    optimizer = FakeOptimizer()
    adapter = QuantumAdapter(optimizer)
    model = object()
    adapter.prepare(model)
    assert optimizer.model is model
    result = adapter.optimize(lambda: 1.25)
    assert result == {"status": "completed", "loss": 1.25}
    assert optimizer.evaluate is not None
