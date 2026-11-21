"""Epoch별 학습 및 선택적 검증 지표를 CSV로 저장한다."""
from __future__ import annotations
import modules.common.TypeContracts as tc

import csv


class HistorySaver:
    FIELDNAMES = ("epoch", "train_loss", "train_acc", "valid_loss", "valid_acc")

    @tc.validate_types
    def __init__(self, enable_saver: bool, path_manager: tc.PathManagerLike) -> None:
        self.path_manager = path_manager
        self.history = []
        self.enable_saver = enable_saver

    @tc.validate_types
    def add(self, epoch: int, epoch_result: tc.Payload) -> None:
        """검증이 없는 학습에서는 valid_* 값을 None으로 유지한다."""
        row = {"epoch": epoch}
        for key in self.FIELDNAMES[1:]:
            value = epoch_result.get(key)
            row[key] = float(value) if value is not None else None
        self.history.append(row)

    @tc.validate_types
    def save(self) -> str:
        if not self.enable_saver:
            return ""
        save_path = self.path_manager.get_history_path()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.FIELDNAMES)
            writer.writeheader()
            writer.writerows(self.history)
        return str(save_path)

    @tc.validate_types
    def get_history(self) -> list[tc.Payload]:
        return list(self.history)

