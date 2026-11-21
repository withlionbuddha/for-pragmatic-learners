from __future__ import annotations
import modules.common.TypeContracts as tc
class EpochResult:
    """
        1 epoch result
    """
    @tc.validate_types
    def __init__(self) -> None:
        self.sum_loss = 0.0
        self.num_batches = 0
        self.total_correct = 0
        self.total_count = 0

    @tc.validate_types
    def add(self, batch_result: tc.Payload) -> None:
        self.sum_loss += batch_result["loss"]
        self.num_batches += 1
        self.total_correct += batch_result["correct"]
        self.total_count += batch_result["total"]

    @tc.validate_types
    def to_train_dict(self, accuracy_calculator: tc.AccuracyAverageLike, loss_calculator: tc.LossAverageLike) -> tc.Payload:
        return {
            "sum_loss": self.sum_loss
            , "num_batches":self.num_batches
            , "total_correct":self.total_correct
            , "total_count" : self.total_count
            , "train_loss": loss_calculator.get_loss_average(self.sum_loss, self.num_batches)
            , "train_acc": accuracy_calculator.get_accuracy_average(self.total_correct, self.total_count)
            , "valid_loss": None
            , "valid_acc": None
            , "checkpoint_loss": loss_calculator.get_loss_average(self.sum_loss, self.num_batches)
        }

    @tc.validate_types
    def to_valid_dict(self, accuracy_calculator: tc.AccuracyAverageLike, loss_calculator: tc.LossAverageLike) -> tc.Payload:
        return {
            "sum_loss": self.sum_loss
            , "num_batches":self.num_batches
            , "total_correct":self.total_correct
            , "total_count" : self.total_count
            , "train_loss": None
            , "train_acc": None
            , "valid_loss": loss_calculator.get_loss_average(self.sum_loss, self.num_batches)
            , "valid_acc": accuracy_calculator.get_accuracy_average(self.total_correct, self.total_count)
            , "checkpoint_loss": loss_calculator.get_loss_average(self.sum_loss, self.num_batches)
        }

