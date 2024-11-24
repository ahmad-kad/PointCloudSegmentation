import argparse
from pathlib import Path
import random
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('data_splitter')

def create_splits(data_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """Create train/val/test splits from processed data."""
    logger = setup_logging()
    
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Get all processed files
    all_scenes = [f.stem for f in data_dir.glob('*.h5')]
    random.shuffle(all_scenes)
    
    # Calculate split sizes
    total_scenes = len(all_scenes)
    train_size = int(total_scenes * train_ratio)
    val_size = int(total_scenes * val_ratio)
    
    # Create splits
    train_scenes = all_scenes[:train_size]
    val_scenes = all_scenes[train_size:train_size + val_size]
    test_scenes = all_scenes[train_size + val_size:]
    
    # Create splits directory
    splits_dir = data_dir / 'splits'
    splits_dir.mkdir(exist_ok=True)
    
    # Save splits
    for split_name, scenes in [
        ('train', train_scenes),
        ('val', val_scenes),
        ('test', test_scenes)
    ]:
        split_file = splits_dir / f'{split_name}.txt'
        with open(split_file, 'w') as f:
            f.write('\n'.join(scenes))
        logger.info(f"Created {split_name} split with {len(scenes)} scenes")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing processed data')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Ratio of training data')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Ratio of validation data')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    random.seed(args.seed)
    create_splits(args.data_dir, args.train_ratio, args.val_ratio)