import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import yaml
from pathlib import Path
import logging
from tqdm import tqdm
from preprocess_data import load_split_scenes
from src.utils.data_utils import DataProcessor
import wandb
from typing import Dict, Tuple


from src.data.scannet_dataset import ScanNetDataset
from src.models.point2net import Point2Net
from src.utils.early_stopping import EarlyStopping
from src.utils.visualization import SegmentationVisualizer
from src.models.losses import InstanceSegLoss, FocalLoss
from src.utils.metrics import InstanceSegmentationMetrics


class TrainingManager:
    """Manages the training process and related components."""
    
    def __init__(self, config: dict, output_dir: Path, logger: logging.Logger):
            self.config = config
            self.output_dir = output_dir
            self.logger = logger
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Initialize metrics and visualization components
            self.setup_metrics()
            self.visualizer = SegmentationVisualizer(output_dir / 'visualizations')
            
            # Initialize early stopping with your existing configuration
            self.early_stopping = EarlyStopping(
                patience=config['training']['early_stopping_patience'],
                mode='max',  # Since we're tracking mIoU
                save_dir=output_dir
            )
            
            # Track metrics history for visualization
            self.train_losses = []
            self.val_losses = []
        
    def setup_metrics(self):
        """Initialize training and validation metrics."""
        self.train_metrics = InstanceSegmentationMetrics(
            threshold=self.config['training'].get('iou_threshold', 0.5)
        )
        self.val_metrics = InstanceSegmentationMetrics(
            threshold=self.config['training'].get('iou_threshold', 0.5)
        )

    # Model initialization
    def create_model(self) -> torch.nn.Module:
        """Initialize the model architecture."""
        model = Point2Net(
            feature_dim=6,
            embedding_dim=self.config['model']['embedding_dim']
        ).to(self.device)
        return model
    
    # Loss function initialization
    def create_criterion(self) -> torch.nn.Module:
        """Initialize the loss function."""
        if self.config['loss']['type'] == 'instance_seg':
            criterion = InstanceSegLoss(
                feature_dim=self.config['model']['embedding_dim'],
                delta_v=self.config['loss']['params']['delta_v'],
                delta_d=self.config['loss']['params']['delta_d']
            ).to(self.device)
        else:
            criterion = FocalLoss(
                alpha=self.config['loss']['params']['focal_alpha'],
                gamma=self.config['loss']['params']['focal_gamma']
            ).to(self.device)
        return criterion
    
    # Optimizer and scheduler initialization
    def create_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """Initialize the optimizer."""
        return optim.Adam(
            model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
    
    def create_scheduler(self, optimizer: torch.optim.Optimizer):
        """Initialize the learning rate scheduler."""
        if self.config['scheduler']['enabled']:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config['scheduler']['T_max'],
                eta_min=self.config['scheduler']['eta_min']
            )
        return None
    
    # Training and validation methods
    def train_epoch(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Execute one training epoch."""
        model.train()
        self.train_metrics.reset()
        
        train_loss = 0
        num_batches = len(train_loader)
        
        with tqdm(train_loader, desc='Training') as pbar:
            for batch in pbar:
                # Move data to device
                features = batch['features'].to(self.device)
                instance_labels = batch['instance_labels'].to(self.device)
                points = features[:, :, :3]
                
                # Forward pass
                optimizer.zero_grad()
                embeddings = model(features)
                
                # Compute loss
                loss_dict = criterion(points, embeddings, instance_labels)
                loss = loss_dict['total_loss'] # Total loss
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.config['training']['gradient_clip']
                )
                optimizer.step()
                
                # Update metrics
                self.train_metrics.update(embeddings, instance_labels)
                metrics = self.train_metrics.get_metrics()
                
                # Update progress
                train_loss += loss.item()
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'miou': f"{metrics['miou']:.4f}"
                })
                
                # Log batch metrics
                self.log_batch_metrics(loss_dict)
        
        # Compute epoch metrics
        epoch_metrics = {
            'loss': train_loss / num_batches,
            **self.train_metrics.get_metrics()
        }
        
        self.train_losses.append(epoch_metrics['loss'])
        return epoch_metrics
    
    # Visualization and early stopping
    def visualize_predictions(
        self,
        epoch: int,
        model: torch.nn.Module,
        val_loader: DataLoader,
        num_samples: int = 5
    ):
        """Integrate your existing visualization code during training."""
        if (epoch + 1) % self.config['training']['visualization_frequency'] == 0:
            try:
                self.logger.info("Generating visualizations...")
                model.eval()
                
                # Visualize predictions for a few samples
                with torch.no_grad():
                    
                    for i, batch in enumerate(val_loader):
                        
                        # Break after visualizing num_samples 
                        if i >= num_samples:
                            break
                            
                        features = batch['features'].to(self.device)
                        instance_labels = batch['instance_labels'].to(self.device)
                        scene_id = batch['scene_id'][0]
                        
                        embeddings = model(features)
                        
                        # Convert to numpy for visualization
                        points = features[0, :, :3].cpu().numpy()
                        colors = features[0, :, 3:].cpu().numpy()
                        embeddings_np = embeddings[0].cpu().numpy()
                        instance_labels_np = instance_labels[0].cpu().numpy()
                        
                        # Use your existing visualization methods
                        self.visualizer.visualize_point_cloud(
                            points,
                            instance_labels_np,
                            f"{scene_id}_epoch{epoch}",
                            original_colors=colors
                        )
                        
                        # Visualize embeddings
                        self.visualizer.visualize_embeddings(
                            embeddings_np,
                            instance_labels_np,
                            f"{scene_id}_epoch{epoch}"
                        )
                
                # Update and save training curves
                self.visualizer.save_metrics_plot(
                    self.train_losses,
                    self.val_losses,
                    filename=f'training_curves_epoch{epoch}'
                )
                
                self.logger.info("Visualizations completed.")
            except Exception as e:
                self.logger.error(f"Visualization failed: {str(e)}")

    # Validation methods
    def validate(self, model, val_loader, criterion):
        """Modified to track losses for visualization."""
        metrics = super().validate(model, val_loader, criterion)
        self.val_losses.append(metrics['val_loss'])
        return metrics
    
    # Logging and checkpointing
    def log_batch_metrics(self, loss_dict: Dict[str, torch.Tensor]):
        """Log batch-level metrics to wandb."""
        for name, value in loss_dict.items():
            wandb.log({f'batch_{name}': value.item()}, commit=False)
    
    # Early stopping
    def check_early_stopping(self, metric_value: float, model: torch.nn.Module, epoch: int) -> bool:
        """Use your existing early stopping implementation."""
        self.early_stopping(metric_value, model, epoch)
        return self.early_stopping.early_stop
    
    # Logging and checkpointing
    def log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: float
    ):
        """Log epoch-level metrics."""
        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'learning_rate': lr,
            **train_metrics,
            **val_metrics
        })
        
        # Log to console
        self.logger.info(
            f"Train - Loss: {train_metrics['loss']:.4f}, "
            f"mIoU: {train_metrics['miou']:.4f}, "
            f"ARI: {train_metrics['ari']:.4f}, "
            f"NMI: {train_metrics['nmi']:.4f}"
        )
        self.logger.info(
            f"Val - Loss: {val_metrics['val_loss']:.4f}, "
            f"mIoU: {val_metrics['val_miou']:.4f}, "
            f"ARI: {val_metrics['val_ari']:.4f}, "
            f"NMI: {val_metrics['val_nmi']:.4f}"
        )
    
    # Checkpointing
    def save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': self.config
        }, self.output_dir / 'best_model.pth')
        
        self.logger.info(f"Saved new best model with validation mIoU: {val_metrics['val_miou']:.4f}")

def setup_logging(output_dir: str) -> logging.Logger:
    # Convert string path to Path object for proper path handling
    log_dir = Path(output_dir)
    
    try:
        # Create output directory if it doesn't exist
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging format with more detailed information
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        
        # Create log file path using Path object's joinpath or / operator
        log_file = log_dir / 'training.log'
        
        # Configure handlers with proper error handling
        file_handler = logging.FileHandler(str(log_file))
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format, date_format))
        
        # Get logger and configure it
        logger = logging.getLogger('training')
        logger.setLevel(logging.INFO)
        
        # Remove any existing handlers to avoid duplicate logging
        logger.handlers.clear()
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Log initial setup message
        logger.info(f"Logging initialized. Log file: {log_file}")
        
        return logger
        
    except OSError as e:
        raise OSError(f"Failed to setup logging in directory {output_dir}: {str(e)}")


def create_train_val_split(data_dir: Path, train_ratio: float = 0.8, seed: int = 42):
    """
    Create train and validation splits from all preprocessed files.
    This function should be called at the start of training.
    
    Args:
        data_dir: Directory containing the preprocessed .h5 files
        train_ratio: Fraction of data to use for training (default 0.8)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple[list, list]: Lists of file paths for training and validation sets
    """
    import numpy as np
    
    # Get all preprocessed files
    h5_files = list(data_dir.glob("*.h5"))
    
    if not h5_files:
        raise RuntimeError(
            f"No preprocessed files (.h5) found in {data_dir}! "
            "Please run preprocessing first."
        )
    
    # Sort files for reproducibility before shuffling
    h5_files.sort()
    
    # Convert to numpy array and shuffle
    h5_files = np.array(h5_files)
    rng = np.random.RandomState(seed)
    rng.shuffle(h5_files)
    
    # Calculate split indices
    split_idx = int(len(h5_files) * train_ratio)
    train_files = h5_files[:split_idx].tolist()
    val_files = h5_files[split_idx:].tolist()
    
    print(f"Created dataset splits:")
    print(f"  - Training: {len(train_files)} files")
    print(f"  - Validation: {len(val_files)} files")
    
    return train_files, val_files

# Add this function to create dataloaders
def create_dataloaders(
    data_dir: str,
    config: dict
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders."""
    data_dir = Path(data_dir)
    
    # Create train/val split
    train_files, val_files = create_train_val_split(
        data_dir,
        train_ratio=config['dataset'].get('train_ratio', 0.8)
    )
    
    # Create datasets
    train_dataset = ScanNetDataset(
        files=train_files,
        config=config,
        num_points=config['dataset'].get('num_points', 16384)
    )
    
    val_dataset = ScanNetDataset(
        files=val_files,
        config=config,
        num_points=config['dataset'].get('num_points', 16384)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )
    
    return train_loader, val_loader

def verify_data_directory(data_dir: Path):
    """Verify the data directory structure and files."""
    logger = logging.getLogger('training')
    
    # Check if directory exists
    if not data_dir.exists():
        raise RuntimeError(f"Data directory not found: {data_dir}")
        
    # List all h5 files
    h5_files = list(data_dir.rglob("*.h5"))
    logger.info(f"Found {len(h5_files)} .h5 files")
    if h5_files:
        logger.info(f"Sample files: {[f.name for f in h5_files[:5]]}")
        
    # Check split files
    splits_dir = data_dir / 'splits'
    if not splits_dir.exists():
        raise RuntimeError(f"Splits directory not found: {splits_dir}")
        
    # Read and verify split files
    for split in ['train', 'val']:
        split_file = splits_dir / f"{split}.txt"
        if split_file.exists():
            with open(split_file, 'r') as f:
                scene_ids = [line.strip() for line in f]
            logger.info(f"{split} split contains {len(scene_ids)} scene IDs")
            if scene_ids:
                logger.info(f"Sample IDs: {scene_ids[:5]}")

def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description='Train Point Cloud Instance Segmentation Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Path to processed data directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory for logs and checkpoints'
    )
    
    args = parser.parse_args()
    
    # Load config and setup
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory and setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Set up logging
    logger = setup_logging(args.output_dir)
    
    try:
        # Initialize wandb
        wandb.init(
            project="scannet-instance-seg",
            config=config,
            dir=str(output_dir)
        )
        
        # Create dataloaders
        train_loader, val_loader = create_dataloaders(args.data_dir, config)
        logger.info("Successfully created data loaders")
        
        # Initialize training components and continue with training
        training_manager = TrainingManager(config=config, output_dir=output_dir, logger=logger)
        model = training_manager.create_model()
        criterion = training_manager.create_criterion()
        optimizer = training_manager.create_optimizer(model)
        scheduler = training_manager.create_scheduler(optimizer)
        
        # Training loop
        best_val_miou = 0.0
        
        for epoch in range(config['training']['num_epochs']):
            logger.info(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
            
            # Train and validate
            train_metrics = training_manager.train_epoch(model, train_loader, criterion, optimizer)
            val_metrics = training_manager.validate(model, val_loader, criterion)
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step(val_metrics['val_miou'])
            
            # Log metrics
            training_manager.log_epoch_metrics(
                epoch,
                train_metrics,
                val_metrics,
                optimizer.param_groups[0]['lr']
            )
            
            # Visualize predictions
            training_manager.visualize_predictions(epoch, model, val_loader)
            
            # Handle model saving and early stopping
            if val_metrics['val_miou'] > best_val_miou:
                best_val_miou = val_metrics['val_miou']
                training_manager.save_checkpoint(epoch, model, optimizer, train_metrics, val_metrics)
            
            # Check early stopping using mIoU
            if training_manager.check_early_stopping(val_metrics['val_miou'], model, epoch):
                logger.info("Early stopping triggered")
                break

        logger.info("Training completed!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()