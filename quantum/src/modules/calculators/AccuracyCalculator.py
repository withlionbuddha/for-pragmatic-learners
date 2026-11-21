from __future__ import annotations
import modules.common.TypeContracts as tc
import torch
from modules.common.CommonModule import CommonModule


class AccuracyCalculator(CommonModule):
    @tc.validate_types
    def __init__(self) -> None:
        super(AccuracyCalculator, self).__init__()

    @tc.validate_types
    def forward(self, predicted_x_batch: tc.torch.Tensor, y_batch: tc.torch.Tensor) -> tc.Metric:
        return self.get_accuracy(predicted_x_batch, y_batch)

    @tc.validate_types
    def calculate(self, correct: int, total: int) -> float:
        """correct와 total을 이용하여 범용 accuracy를 계산한다."""
        return correct / total if total > 0 else 0.0

    @tc.validate_types
    def get_accuracy(self, predicted_x_batch: tc.torch.Tensor, y_batch: tc.torch.Tensor) -> tc.Metric:
        """전체 target을 대상으로 accuracy를 계산한다."""
        with torch.no_grad():
            pred_x_batch_cpu = predicted_x_batch.detach().cpu()
            y_batch_cpu = y_batch.detach().cpu()

            pred_ids = pred_x_batch_cpu.argmax(dim=-1)

            correct = (pred_ids == y_batch_cpu).sum().item()
            total = y_batch_cpu.numel()
            accuracy = self.calculate(correct, total)

        return accuracy, correct, total

    @tc.validate_types
    def get_accuracy_average(self, total_correct: int, total_count: int) -> float:
        return self.calculate(total_correct, total_count)

