from __future__ import annotations
import modules.common.TypeContracts as tc
import time

class EpochLogger:
    @tc.validate_types
    def __init__(self) -> None:
        self.start_time = time.perf_counter()
  
    @tc.validate_types
    def _is_not_logging(self, epoch: int, log_interval: int) -> bool :
        return epoch % log_interval != 0

    @tc.validate_types
    def print_epoch_result(self, epoch: int, log_interval: int, epoch_result: tc.Payload) -> None:
        if self._is_not_logging(epoch=epoch, log_interval=log_interval):
             return

        message = f"epoch={epoch:04d}"

        if epoch_result["train_loss"] is not None:
            message += f", train_loss={epoch_result['train_loss']:.4f}"
            message += f", train_acc={epoch_result['train_acc']:.4f}"
  
        if epoch_result["valid_loss"] is not None:
            message += f", valid_loss={epoch_result['valid_loss']:.4f}"
            message += f", valid_acc={epoch_result['valid_acc']:.4f}"

        message += f", elapsed={self.get_elapsed_min():.2f} min"

        print(message)

    @tc.validate_types
    def get_elapsed_min(self) -> float :
        return (time.perf_counter() - self.start_time) / 60

