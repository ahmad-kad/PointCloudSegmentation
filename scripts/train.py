import torch
import argparse
import yaml
from pathlib import Path
from src.training.trainer import Trainer
from src.data import ScanNetDataModule
from src.utils.logger import Logger
import sys
import logging

def parse_args():
    parser = argparse.ArgumentParser(description='Train instance segmentation model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model', type=str, choices=['pointnet2', 'sparsecnn'], required=True)
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--exp-name', type=str, default='default', help='Experiment name')
    return parser.parse_args()

def setup_logging(output_dir: str):
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(output_dir) / 'training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Update config with experiment name
    config['experiment']['name'] = args.exp_name
    
    # Create output directory
    output_dir = Path(config['training']['checkpointing']['save_dir']) / args.exp_name
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Set up logging
    setup_logging(str(output_dir))
    logger = logging.getLogger('training')
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
        
    # Initialize trainer
    trainer = Trainer(args.config)
    
    # Initialize data module
    data_module = ScanNetDataModule(
        data_dir=config['dataset']['root_dir'],
        config_path=args.config,
        batch_size=config['training']['batch_size']['train'],
        num_workers=config['hardware']['num_workers']
    )
    
    # Set up data
    try:
        data_module.setup(stage='fit')
        logger.info("Data module initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize data module: {str(e)}")
        sys.exit(1)
    
    # Initialize model
    try:
        model = trainer.setup_model(args.model)
        logger.info(f"Initialized {args.model} model")
        
        # Load checkpoint if resuming
        if args.resume:
            epoch, metrics = trainer.load_checkpoint(
                model,
                args.resume
            )
            logger.info(f"Resumed from checkpoint at epoch {epoch}")
            logger.info(f"Previous metrics: {metrics}")
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        sys.exit(1)
        
    # Set up wandb logging
    if config['logging']['wandb']['enabled']:
        import wandb
        wandb.init(
            project=config['logging']['wandb']['project'],
            entity=config['logging']['wandb']['entity'],
            name=args.exp_name,
            config=config,
            resume=True if args.resume else False
        )
        logger.info("Initialized W&B logging")
    
    # Training loop
    try:
        model = trainer.train(
            model=model,
            train_loader=data_module.train_dataloader(),
            val_loader=data_module.val_dataloader(),
            model_name=args.model
        )
        logger.info("Training completed successfully")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)
    finally:
        # Clean up
        if config['logging']['wandb']['enabled']:
            wandb.finish()
        
        # Save final model
        try:
            final_checkpoint_path = output_dir / 'final_model.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config
            }, final_checkpoint_path)
            logger.info(f"Saved final model to {final_checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save final model: {str(e)}")

def validate_config(config: dict) -> bool:
    """Validate configuration file."""
    required_keys = [
        'dataset',
        'model',
        'training',
        'optimizer',
        'hardware',
        'logging'
    ]
    
    for key in required_keys:
        if key not in config:
            logging.error(f"Missing required config key: {key}")
            return False
            
    # Validate specific parameters
    if config['training']['batch_size']['train'] <= 0:
        logging.error("Batch size must be positive")
        return False
        
    if config['optimizer']['learning_rate'] <= 0:
        logging.error("Learning rate must be positive")
        return False
        
    return True

def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    args = parse_args()
    
    # Load and validate config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    if not validate_config(config):
        sys.exit(1)
        
    # Set random seed for reproducibility
    set_random_seed(config['training']['seed'])
    
    # Check hardware requirements
    if config['hardware']['device'] == 'cuda' and not torch.cuda.is_available():
        logging.error("CUDA device requested but not available. Falling back to CPU.")
        config['hardware']['device'] = 'cpu'
        
    # Apple Silicon specific setup
    if config['hardware']['device'] == 'mps':
        if not torch.backends.mps.is_available():
            logging.error("MPS device requested but not available. Falling back to CPU.")
            config['hardware']['device'] = 'cpu'
        else:
            # Enable Metal Performance Shaders
            torch.backends.mps.enable_mps()
    
    main()