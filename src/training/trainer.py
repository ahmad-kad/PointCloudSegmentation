import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from pathlib import Path
from typing import Dict, Optional, List
import yaml
from tqdm import tqdm
import numpy as np

from ..models import PointNet2InstanceSegmentation, SparseCNNInstanceSegmentation
from ..evaluation import evaluate_predictions
from .losses import InstanceSegLoss

class Trainer:
    """Training manager for instance segmentation models."""
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device(self.config['hardware']['device'])
        self.setup_wandb()
        
    def setup_wandb(self):
        """Initialize Weights & Biases logging."""
        if self.config['logging']['wandb']['enabled']:
            wandb.init(
                project=self.config['logging']['wandb']['project'],
                entity=self.config['logging']['wandb']['entity'],
                config=self.config
            )
            
    def setup_model(self, model_name: str) -> nn.Module:
        """Initialize model architecture."""
        if model_name == 'pointnet2':
            with open('configs/model_configs/pointnet.yaml', 'r') as f:
                model_config = yaml.safe_load(f)
            model = PointNet2InstanceSegmentation(model_config['model'])
        elif model_name == 'sparsecnn':
            with open('configs/model_configs/sparsecnn.yaml', 'r') as f:
                model_config = yaml.safe_load(f)
            model = SparseCNNInstanceSegmentation(model_config['model'])
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        return model.to(self.device)
    
    def train(self, 
              model: nn.Module,
              train_loader: torch.utils.data.DataLoader,
              val_loader: torch.utils.data.DataLoader,
              model_name: str) -> nn.Module:
        """Train the model."""
        # Setup loss and optimizer
        criterion = InstanceSegLoss(self.config['training'])
        optimizer = self.setup_optimizer(model)
        scheduler = self.setup_scheduler(optimizer)
        
        # Training loop
        best_val_score = 0.0
        num_epochs = self.config['training']['num_epochs']
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_losses = []
            train_metrics = []
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
                # Move data to device
                batch = {k: v.to(self.device) if v is not None else None 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = model(batch['points'], batch.get('features'))
                loss, loss_dict = criterion(outputs, batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                if self.config['training']['gradient_clip_val'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config['training']['gradient_clip_val']
                    )
                
                optimizer.step()
                
                # Log metrics
                train_losses.append(loss_dict)
                metrics = self.compute_metrics(outputs, batch)
                train_metrics.append(metrics)
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'mAP': metrics['mAP']
                })
            
            # Validation
            if (epoch + 1) % self.config['training']['eval_interval'] == 0:
                val_metrics = self.validate(model, val_loader, criterion)
                
                # Log to wandb
                if self.config['logging']['wandb']['enabled']:
                    wandb.log({
                        'epoch': epoch + 1,
                        'train_loss': np.mean([l['total_loss'] for l in train_losses]),
                        'val_loss': val_metrics['loss'],
                        'train_mAP': np.mean([m['mAP'] for m in train_metrics]),
                        'val_mAP': val_metrics['mAP']
                    })
                    
                # Save checkpoint
                if val_metrics['mAP'] > best_val_score:
                    best_val_score = val_metrics['mAP']
                    self.save_checkpoint(
                        model,
                        optimizer,
                        epoch,
                        val_metrics,
                        model_name,
                        is_best=True
                    )
                    
            # Update scheduler
            scheduler.step()
            
        return model
    
    def validate(self,
                model: nn.Module,
                val_loader: torch.utils.data.DataLoader,
                criterion: InstanceSegLoss) -> Dict:
        """Validate the model."""
        model.eval()
        val_losses = []
        val_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                batch = {k: v.to(self.device) if v is not None else None 
                        for k, v in batch.items()}
                
                outputs = model(batch['points'], batch.get('features'))
                loss, loss_dict = criterion(outputs, batch)
                
                val_losses.append(loss_dict)
                metrics = self.compute_metrics(outputs, batch)
                val_metrics.append(metrics)
                
        # Aggregate metrics
        mean_metrics = {
            'loss': np.mean([l['total_loss'] for l in val_losses]),
            'mAP': np.mean([m['mAP'] for m in val_metrics]),
            'IoU': np.mean([m['IoU'] for m in val_metrics])
        }
        
        return mean_metrics
    
    def setup_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Initialize optimizer."""
        config = self.config['optimizer']
        
        if config['type'].lower() == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {config['type']}")
            
        return optimizer
    
    def setup_scheduler(self, 
                       optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        """Initialize learning rate scheduler."""
        config = self.config['optimizer']['scheduler']
        
        if config['type'].lower() == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config['T_max'],
                eta_min=config['eta_min']
            )
        elif config['type'].lower() == 'one_cycle':
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config['max_lr'],
                epochs=config['epochs'],
                steps_per_epoch=config['steps_per_epoch']
            )
        else:
            raise ValueError(f"Unknown scheduler: {config['type']}")
            
        return scheduler
    
    def compute_metrics(self, outputs: Dict, targets: Dict) -> Dict:
        """Compute evaluation metrics."""
        metrics = evaluate_predictions(
            outputs,
            targets,
            self.config['evaluation']
        )
        return metrics
    
    def save_checkpoint(self,
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       metrics: Dict,
                       model_name: str,
                       is_best: bool = False):
        """Save model checkpoint."""
        save_dir = Path(self.config['training']['checkpointing']['save_dir'])
        save_dir.mkdir(exist_ok=True, parents=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save latest checkpoint
        torch.save(
            checkpoint,
            save_dir / f"{model_name}_latest.pth"
        )
        
        # Save best checkpoint
        if is_best:
            torch.save(
                checkpoint,
                save_dir / f"{model_name}_best.pth"
            )
            
        # Save periodic checkpoint
        if self.config['training']['checkpointing']['save_interval_epochs'] > 0:
            if (epoch + 1) % self.config['training']['checkpointing']['save_interval_epochs'] == 0:
                torch.save(
                    checkpoint,
                    save_dir / f"{model_name}_epoch{epoch+1}.pth"
                )
                
    def load_checkpoint(self,
                       model: nn.Module,
                       checkpoint_path: str,
                       optimizer: Optional[torch.optim.Optimizer] = None) -> tuple:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        return checkpoint['epoch'], checkpoint['metrics']