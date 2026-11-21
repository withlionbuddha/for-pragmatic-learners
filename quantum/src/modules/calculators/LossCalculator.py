from __future__ import annotations
import modules.common.TypeContracts as tc
from modules.common.CommonModule import CommonModule

class LossCalculator(CommonModule):
    @tc.validate_types
    def __init__(self, criterion: tc.Callable[..., tc.torch.Tensor] | None) -> None:
        super(LossCalculator, self).__init__()
        self.criterion = criterion

    @tc.validate_types
    def forward(self, predicted_x_batch: tc.torch.Tensor, y_batch: tc.torch.Tensor) -> tc.torch.Tensor:
        return self.get_loss(predicted_x_batch, y_batch)
    
    @tc.validate_types
    def get_loss(self, predicted_x_batch: tc.torch.Tensor, y_batch: tc.torch.Tensor) -> tc.torch.Tensor:
        """
        """
        if self.criterion is None:
            raise RuntimeError("criterion이 없습니다. set_criterion()을 먼저 실행해야 합니다.")
        
        return self.criterion(
                predicted_x_batch.reshape(-1, predicted_x_batch.size(-1)),
                y_batch.reshape(-1),
            )
    
    @tc.validate_types
    def get_loss_average(self, sum_loss: float, num_batches: int) -> float:
        return sum_loss / num_batches if num_batches > 0 else 0.0

