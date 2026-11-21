from __future__ import annotations
import modules.common.TypeContracts as tc
from pathlib import Path
from datetime import datetime
from uuid import uuid4

class FilePathManager:

    @tc.validate_types
    def __init__(self, root_dir: tc.PathValue | None=None, run_id: str | None=None) -> None:

        if root_dir is None:
            root_dir = (
                Path(__file__).resolve().parents[3]
                / "models"
            )

        self.root_dir = Path(root_dir)

        if run_id is None:
            run_id = (
                "train_"
                + datetime.now().strftime("%Y%m%d_%H%M%S")
                + "_"
                + uuid4().hex[:8]
            )

        self.run_dir = self.root_dir / run_id

        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.history_dir = self.run_dir / "history"

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=False
        )

        self.history_dir.mkdir(
            parents=True,
            exist_ok=False
        )

    @tc.validate_types
    def get_run_dir(self) -> tc.PathValue:
        return self.run_dir

    @tc.validate_types
    def get_checkpoint_dir(self) -> tc.PathValue:
        return self.checkpoint_dir

    @tc.validate_types
    def get_history_dir(self) -> tc.PathValue:
        return self.history_dir

    @tc.validate_types
    def get_checkpoint_path(self, epoch: int, loss: float) -> tc.PathValue:
        filename = f"best_model_{epoch}_{loss:.6f}.pth"
        return self.checkpoint_dir / filename

    @tc.validate_types
    def get_best_checkpoint_path(self, loss: float | None=None) -> tc.PathValue | None:
        """Loss가 지정되면 해당 Checkpoint, 없으면 최소 Loss 파일을 반환한다."""
        checkpoint_files = list(self.checkpoint_dir.glob("best_model_*.pth"))

        if not checkpoint_files:
            raise FileNotFoundError(
                f"Checkpoint 파일이 없습니다: {self.checkpoint_dir}"
            )

        if loss is None:
            return min(
                checkpoint_files,
                key=lambda path: float(path.stem.rsplit("_", 1)[-1])
            )

        return self._find_checkpoint_by_loss(
            checkpoint_files=checkpoint_files,
            loss=loss,
        )

    @tc.validate_types
    def _find_checkpoint_by_loss(self, checkpoint_files: list[tc.PathValue], loss: float) -> tc.PathValue:
        """지정된 Loss에 해당하는 Checkpoint 파일을 반환한다."""
        target_loss = f"{loss:.6f}"

        for path in checkpoint_files:
            checkpoint_loss = path.stem.rsplit("_", 1)[-1]
            if checkpoint_loss == target_loss:
                return path

        raise FileNotFoundError(
            f"Loss={loss:.6f}에 해당하는 Checkpoint가 없습니다: "
            f"{self.checkpoint_dir}"
        )

    @tc.validate_types
    def get_history_path(self) -> tc.PathValue:
        return self.history_dir / "history.csv"

    @tc.validate_types
    def get_metadata_path(self) -> tc.PathValue:
        return self.run_dir / "metadata.json"
