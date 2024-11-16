import argparse
import yaml
from pathlib import Path
from src.utils.data_utils import DataProcessor
import logging

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess ScanNet dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--raw-dir', type=str, required=True, help='Path to raw data')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--split', type=str, default='all', choices=['train', 'val', 'test', 'all'])
    return parser.parse_args()

def setup_logging(output_dir: str):
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(output_dir) / 'preprocessing.log'),
            logging.StreamHandler()
        ]
    )

def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Set up logging
    setup_logging(args.output_dir)
    logger = logging.getLogger('preprocessing')
    
    # Initialize data processor
    processor = DataProcessor(args.config)
    
    # Process each split
    splits = ['train', 'val', 'test'] if args.split == 'all' else [args.split]
    
    for split in splits:
        logger.info(f"Processing {split} split...")
        
        # Create output directory for split
        split_output_dir = Path(args.output_dir) / split
        split_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Process data
        processor.preprocess_dataset(
            raw_data_dir=Path(args.raw_dir) / split,
            output_dir=split_output_dir,
            split=split
        )
        
        logger.info(f"Completed processing {split} split")
        
    logger.info("Preprocessing completed successfully")

if __name__ == '__main__':
    main()