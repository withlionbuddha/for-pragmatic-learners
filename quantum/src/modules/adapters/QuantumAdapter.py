class QuantumAdapter:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def prepare(self, model):
        self.optimizer.initialize_parameters(model)

    def optimize(self, evaluate):
        result = self.optimizer.run_objective(evaluate)
        return result
