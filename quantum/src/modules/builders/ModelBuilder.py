from __future__ import annotations
import modules.common.TypeContracts as tc
import torch.nn as nn

from modules.common.CommonModule import CommonModule

class SequentialModel(CommonModule):
    """ModelBuilder가 생성하는 Sequential 모델."""

    @tc.validate_types
    def __init__(self, layers: list[tc.nn.Module], device_name: tc.DeviceValue | None=None) -> None:
        super().__init__(device_name=device_name)
        self.network = nn.Sequential(*layers)
        self.move_to_device()

    @tc.validate_types
    def forward(self, x: tc.torch.Tensor) -> tc.torch.Tensor:
        return self.network(x)


class ModelBuilder:
    """PyTorch layer를 순서대로 조립하여 모델을 생성한다."""

    @tc.validate_types
    def __init__(self, device_name: tc.DeviceValue | None=None) -> None:
        self.device_name = device_name
        self.layers = []

    @tc.validate_types
    def add(self, layer: tc.nn.Module | None) -> ModelBuilder:
        if layer is None:
            raise ValueError("layer가 없습니다.")
        self.layers.append(layer)
        return self

    @tc.validate_types
    def build(self) -> SequentialModel:
        if not self.layers:
            raise ValueError("layer가 없습니다.")
        return SequentialModel(
            layers=self.layers,
            device_name=self.device_name,
        )

