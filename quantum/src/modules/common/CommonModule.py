from __future__ import annotations
import modules.common.TypeContracts as tc
import torch.nn as nn
import torch

class CommonModule(nn.Module):
    @tc.validate_types
    def __init__(self, device_name: tc.DeviceValue | None=None) -> None:
        super(CommonModule, self).__init__()
        self.device = self.set_device(device_name)

    @tc.validate_types
    def set_device(self, device_name: tc.DeviceValue) -> tc.torch.device :
        device = None
        if device_name is None:
            if hasattr(torch, "xpu") and torch.xpu.is_available():
               device =torch.device("xpu")
            elif hasattr(torch, "cuda") and torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device =torch.device("cpu")
        else:
            device = torch.device(device_name)
        return device
        
    @tc.validate_types
    def move_to_device(self) -> tc.nn.Module:
        """
            nn.Parameter, nn.Module, register_buffer 을 xpu memory로 옮김.
        """
        return self.to(self.device)

    @tc.validate_types
    def get_device(self) -> tc.torch.device:
        return self.device #next(self.parameters()).device
    
    


