from __future__ import annotations
import modules.common.TypeContracts as tc
class EarlyStopping:
    @tc.validate_types
    def __init__(self, patience: int=0) -> None:
        self.patience = patience
        self.patience_counter = 0 

    @tc.validate_types
    def is_early_stopping(self) -> bool:
        return self.patience_counter >= self.patience
    
    @tc.validate_types
    def get_patience(self) -> int:
        return self.patience
    
    @tc.validate_types
    def get_patience_counter(self) -> int:
        return self.patience_counter
    
    @tc.validate_types
    def update(self, is_improved: bool, current_loss: float) -> None :
        if is_improved :
            self.patience_counter = 0
        else :
            self.patience_counter += 1
            print(f"{current_loss} 개선 없음: {self.patience_counter} / {self.patience}")

