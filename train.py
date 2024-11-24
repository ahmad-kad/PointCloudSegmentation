import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import yaml
from pathlib import Path
import logging
from tqdm import tqdm
import wandb
import numpy as np

from src.data.scannet_dataset import ScanNetDataset
from src.models.point2net import Point2Net, DiscriminativeLoss
from src.utils.early_stopping import EarlyStopping
from src.utils.visualization import SegmentationVisualizer
from src.models.losses import InstanceSegLoss, PointInstanceLoss, FocalLoss

def visualize_predictions(
    model: torch.nn.Module,
    data_loader: DataLoader,
    visualizer: SegmentationVisualizer,
    device: torch.device,
    num_samples: int = 5
):
    """Visualize predictions for a few samples."""
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_samples:
                break
                
            features = batch['features'].to(device)
            instance_labels = batch['instance_labels'].to(device)
            scene_id = batch['scene_id'][0]  # Assuming batch size 1 for visualization
            
            # Get embeddings
            embeddings = model(features)
            
            # Convert to numpy for visualization
            points = features[0, :, :3].cpu().numpy()
            colors = features[0, :, 3:].cpu().numpy()
            embeddings = embeddings[0].cpu().numpy()
            instance_labels = instance_labels[0].cpu().numpy()
            
            # Visualize point cloud
            visualizer.visualize_point_cloud(
                points,
                instance_labels,
                f"{scene_id}",
                original_colors=colors
            )
            
            # Visualize embeddings
            visualizer.visualize_embeddings(
                embeddings,
                instance_labels,
                f"{scene_id}"
            )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to processed data')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory')
    args = parser.parse_args()
    
    # Load config and setup
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    visualizer = SegmentationVisualizer(output_dir / 'visualizations')
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('training')
    
    # Initialize wandb
    wandb.init(project="scannet-instance-seg", config=config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets and dataloaders
    train_dataset = ScanNetDataset(
        root_dir=args.data_dir,
        split='train',
        config_path=args.config
    )
    
    val_dataset = ScanNetDataset(
        root_dir=args.data_dir,
        split='val',
        config_path=args.config
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    # Create model and training components
    model = Point2Net(
        feature_dim=6,
        embedding_dim=config['model']['embedding_dim']
    ).to(device)
    
    # Initialize criterion
    if config['loss']['type'] == 'instance_seg':
        criterion = InstanceSegLoss(
            feature_dim=config['model']['embedding_dim'],
            delta_v=config['loss']['params']['delta_v'],
            delta_d=config['loss']['params']['delta_d']
        ).to(device)
    else:
        criterion = FocalLoss(
            alpha=config['loss']['params']['focal_alpha'],
            gamma=config['loss']['params']['focal_gamma']
        ).to(device)

    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create scheduler here, before the training loop
    if config['scheduler']['enabled']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['scheduler']['T_max'],
            eta_min=config['scheduler']['eta_min']
        )
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience'],
        save_dir=output_dir
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['num_epochs']):
        logger.info(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        
        # Train
        model.train()
        train_loss = 0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc='Training'):
            features = batch['features'].to(device)
            instance_labels = batch['instance_labels'].to(device)
            points = features[:, :, :3]
            
            optimizer.zero_grad()
            embeddings = model(features)
            
            if config['loss']['type'] == 'instance_seg':
                loss_dict = criterion(points, embeddings, instance_labels)
                loss = loss_dict['total_loss']
                
                # Log detailed losses
                for loss_name, loss_value in loss_dict.items():
                    wandb.log({
                        f'train_{loss_name}': loss_value.item()
                    }, commit=False)
            else:
                loss = criterion(embeddings, instance_labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            # Log batch loss
            wandb.log({
                'batch_loss': loss.item(),
                'epoch': epoch
            })
        
        avg_train_loss = train_loss / num_batches
        train_losses.append(avg_train_loss)
        logger.info(f"Training loss: {avg_train_loss:.4f}")
        
        # Validate
        model.eval()
        val_loss = 0  # Initialize val_loss here
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                features = batch['features'].to(device)
                instance_labels = batch['instance_labels'].to(device)
                points = features[:, :, :3]
                
                embeddings = model(features)
                
                if config['loss']['type'] == 'instance_seg':
                    loss_dict = criterion(points, embeddings, instance_labels)
                    loss = loss_dict['total_loss']
                    
                    # Log validation loss components
                    for loss_name, loss_value in loss_dict.items():
                        wandb.log({
                            f'val_{loss_name}': loss_value.item()
                        }, commit=False)
                else:
                    loss = criterion(embeddings, instance_labels)
                
                val_loss += loss.item()
                num_val_batches += 1
        
        # Calculate average validation loss
        avg_val_loss = val_loss / num_val_batches
        val_losses.append(avg_val_loss)
        logger.info(f"Validation loss: {avg_val_loss:.4f}")
        
        # Update learning rate scheduler
        if config['scheduler']['enabled']:
            scheduler.step()
        
        # Log detailed losses if using instance segmentation loss
        if isinstance(loss_dict, dict):
            logger.info(f"Detailed losses - "
                      f"Instance: {loss_dict['instance_loss']:.4f}, "
                      f"Smoothness: {loss_dict['smoothness_loss']:.4f}, "
                      f"Pull: {loss_dict['pull_loss']:.4f}, "
                      f"Push: {loss_dict['push_loss']:.4f}")
        
        # Log epoch metrics
        wandb.log({
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'epoch': epoch,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Visualize predictions periodically
        try:
            # Visualize predictions periodically
            if (epoch + 1) % config['training']['visualization_frequency'] == 0:
                logger.info("Generating visualizations...")
                visualize_predictions(
                    model,
                    val_loader,
                    visualizer,
                    device,
                    num_samples=5
                )
                logger.info("Visualizations completed.")
        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")
            # Continue training even if visualization fails
            pass
        
        # Update training curves plot
        try:
            # Update training curves plot
            visualizer.save_metrics_plot(train_losses, val_losses)
        except Exception as e:
            logger.error(f"Metrics plot failed: {str(e)}")
            pass
        
        # Early stopping
        early_stopping(avg_val_loss, model, epoch)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config
            }, output_dir / 'best_model.pth')
            logger.info(f"Saved new best model with validation loss: {avg_val_loss:.4f}")
    
    
    # Final visualization
    logger.info("Creating final visualizations...")
    visualize_predictions(
        model,
        val_loader,
        visualizer,
        device,
        num_samples=10
    )
    
    logger.info("Training completed!")

if __name__ == '__main__':
    main()