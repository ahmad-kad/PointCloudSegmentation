import logging
import sys
from pathlib import Path
from typing import Optional
import wandb
import torch
from datetime import datetime

class Logger:
    """Custom logger for point cloud instance segmentation."""
    
    def __init__(self,
                 name: str,
                 save_dir: str,
                 wandb_config: Optional[dict] = None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup file logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.save_dir / f"{name}.log")
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        # Setup W&B logging
        self.wandb_enabled = wandb_config is not None
        if self.wandb_enabled:
            wandb.init(
                project=wandb_config['project'],
                entity=wandb_config['entity'],
                config=wandb_config,
                name=f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
    def log_metrics(self, metrics: dict, step: int):
        """Log metrics to file and W&B."""
        # Format metrics string
        metrics_str = ' | '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step} | {metrics_str}")
        
        # Log to W&B
        if self.wandb_enabled:
            wandb.log(metrics, step=step)
            
    def log_model(self, model: torch.nn.Module, step: int):
        """Save model checkpoint and log to W&B."""
        checkpoint_dir = self.save_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"model_step{step}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        
        if self.wandb_enabled:
            artifact = wandb.Artifact(
                name=f"model-step{step}",
                type='model',
                description=f"Model checkpoint at step {step}"
            )
            artifact.add_file(str(checkpoint_path))
            wandb.log_artifact(artifact)
            
    def log_config(self, config: dict):
        """Log configuration."""
        self.logger.info("Configuration:")
        for k, v in config.items():
            self.logger.info(f"{k}: {v}")
            
    def log_images(self, 
                  images: dict,
                  step: int,
                  save_local: bool = True):
        """Log images to W&B and optionally save locally."""
        if save_local:
            image_dir = self.save_dir / 'images'
            image_dir.mkdir(exist_ok=True)
            
            for name, img in images.items():
                save_path = image_dir / f"{name}_step{step}.png"
                img.save(str(save_path))
                
        if self.wandb_enabled:
            wandb.log(images, step=step)
            
    def log_pr_curve(self,
                     name: str,
                     labels: torch.Tensor,
                     predictions: torch.Tensor,
                     step: int):
        """Log precision-recall curve to W&B."""
        if self.wandb_enabled:
            wandb.log({
                name: wandb.plot.pr_curve(
                    labels.cpu().numpy(),
                    predictions.cpu().numpy(),
                    classes=['background'] + [f'class_{i}' for i in range(predictions.shape[1]-1)]
                )
            }, step=step)
            
    def finish(self):
        """Clean up logging."""
        if self.wandb_enabled:
            wandb.finish()
            
    def log_error(self, error: Exception):
        """Log error message."""
        self.logger.error(str(error), exc_info=True)