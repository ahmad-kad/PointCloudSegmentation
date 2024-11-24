import numpy as np
import torch
from pathlib import Path

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0,
        mode: str = 'min',
        save_dir: Path = None
    ):
        """
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for loss, 'max' for metrics like accuracy
            save_dir: Directory to save the best model
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.save_dir = Path(save_dir) if save_dir else None
        self.best_score = None
        self.counter = 0
        self.best_model = None
        self.early_stop = False
        
        self.save_dir.mkdir(parents=True, exist_ok=True) if save_dir else None
    
    def __call__(self, score, model, epoch):
        if self.mode == 'min':
            score = -score
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, score, epoch)
        elif score <= self.best_score + self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, score, epoch)
            self.counter = 0
    
    def save_checkpoint(self, model, score, epoch):
        """Save model when validation loss decreases."""
        if self.save_dir is None:
            return
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'score': score
        }, self.save_dir / 'best_model.pth')