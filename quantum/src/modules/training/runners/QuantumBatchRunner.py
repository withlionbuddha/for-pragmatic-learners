from __future__ import annotations

from contextlib import nullcontext

import torch

import modules.common.TypeContracts as tc
from modules.training.profilers.TrainingProfiler import TrainingProfiler

"""
    1 batch 당 train 과 valid 실행
"""


class QuantumBatchRunner:
    requires_grad = False

    @tc.validate_types
    def __init__(
        self,
        model: tc.nn.Module | tc.DeviceModel,
        loss_calculator: tc.Callable[..., tc.torch.Tensor],
        accuracy_calculator: tc.Callable[..., tc.Metric],
    ) -> None:
        self.model = model
        self.loss_calculator = loss_calculator
        self.accuracy_calculator = accuracy_calculator

    @tc.validate_types
    def run_train_batch(
        self,
        x_batch: object,
        y_batch: tc.torch.Tensor,
        optimizer: tc.QuantumOptimizerLike,
        profiler: TrainingProfiler | None = None,
    ) -> tc.Payload:
        x_batch, y_batch = self._move_batch_to_device(x_batch, y_batch)

        optimizer.prepare(self.model)

        @tc.validate_types
        def objective_function() -> float:
            """
            목적함수 : 모델의 Forward와 Loss 계산을 수행하고, Optimizer에 손실값을 반환.
            """
            with torch.no_grad():
                with (
                    profiler("objective_evaluation")
                    if profiler is not None
                    else nullcontext()
                ):
                    predicted = self.model(x_batch)
                    loss = self.loss_calculator(
                        predicted,
                        y_batch,
                    )
                    return float(loss.item())

        with profiler("optimizer_step") if profiler is not None else nullcontext():
            optimizer_result = optimizer.optimize(objective_function)

        with torch.no_grad():
            with (
                profiler("training_monitor") if profiler is not None else nullcontext()
            ):
                predicted_batch = self.model(x_batch)

        with torch.no_grad():
            loss_batch = self.loss_calculator(predicted_batch, y_batch)
            accuracy, correct, total = self.accuracy_calculator(
                predicted_batch,
                y_batch,
            )

        return {
            "loss": loss_batch.item(),
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "optimizer_result": optimizer_result,
        }

    @tc.validate_types
    def run_valid_batch(
        self, x_batch: object, y_batch: tc.torch.Tensor, profiler: TrainingProfiler
    ) -> tc.Payload:
        x_batch, y_batch = self._move_batch_to_device(x_batch, y_batch)

        with torch.no_grad():
            with profiler("validation") if profiler is not None else nullcontext():
                predicted_batch = self.model(x_batch)
            loss_batch = self.loss_calculator(predicted_batch, y_batch)

            accuracy, correct, total = self.accuracy_calculator(
                predicted_batch,
                y_batch,
            )

            return {
                "loss": loss_batch.item(),
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
            }

    @tc.validate_types
    def _move_batch_to_device(
        self, x_batch: object, y_batch: tc.torch.Tensor
    ) -> tuple[object, object]:
        device = self.model.get_device()
        return (
            self._move_to_device(x_batch, device),
            self._move_to_device(y_batch, device),
        )

    @tc.validate_types
    def _move_to_device(self, data: object, device: tc.DeviceValue) -> object:
        if torch.is_tensor(data):
            return data.to(device)

        if isinstance(data, dict):
            return {
                key: self._move_to_device(value, device) for key, value in data.items()
            }

        if isinstance(data, tuple):
            return tuple(self._move_to_device(value, device) for value in data)

        if isinstance(data, list):
            return [self._move_to_device(value, device) for value in data]

        return data

    @tc.validate_types
    def get_loss_calculator(self) -> tc.Callable[..., tc.torch.Tensor]:
        return self.loss_calculator

    @tc.validate_types
    def get_accuracy_calculator(self) -> tc.Callable[..., tc.Metric]:
        return self.accuracy_calculator
