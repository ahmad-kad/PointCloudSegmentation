import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import argparse
import yaml
from src.utils.data_utils import DataProcessor
import logging
from tqdm import tqdm
import shutil

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

def load_split_scenes(split_file: str) -> list:
    """Load scene IDs from a split file."""
    if not Path(split_file).exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    with open(split_file, 'r') as f:
        return [line.strip() for line in f]

def copy_split_files(output_dir: Path):
    """Copy split files to output directory."""
    splits_dir = output_dir / 'splits'
    splits_dir.mkdir(exist_ok=True, parents=True)
    
    # Copy split files
    for split in ['train', 'val', 'test']:
        src_file = Path(f'{split}.txt')
        if src_file.exists():
            shutil.copy2(src_file, splits_dir / f'{split}.txt')
            logging.info(f"Copied {split}.txt to splits directory")
        else:
            logging.error(f"Split file not found: {src_file}")

def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Set up logging
    setup_logging(args.output_dir)
    logger = logging.getLogger('preprocessing')
    
    # Initialize data processor
    processor = DataProcessor(args.config)
    
    # Process each split
    splits = ['train', 'val', 'test'] if args.split == 'all' else [args.split]
    
    raw_dir = Path(args.raw_dir)
    logger.info(f"Raw data directory: {raw_dir}")
    
    # Load all scenes to process
    scenes_to_process = set()
    for split in splits:
        try:
            split_scenes = load_split_scenes(f'{split}.txt')
            scenes_to_process.update(split_scenes)
            logger.info(f"Found {len(split_scenes)} scenes in {split} split")
        except FileNotFoundError as e:
            logger.error(str(e))
            continue
    
    logger.info(f"Total unique scenes to process: {len(scenes_to_process)}")
    
    # Point to the correct scans directory
    scans_dir = raw_dir / 'scannet' / 'scans'
    if not scans_dir.exists():
        logger.info("Checking if raw_dir itself contains the scans...")
        
        # Check if the raw_dir directly contains the scene folders
        if any(d.name.startswith('scene') for d in raw_dir.iterdir() if d.is_dir()):
            scans_dir = raw_dir
            logger.info(f"Found scene folders in {raw_dir}")
        else:
            logger.error("No scene folders found. Please check the raw_dir path.")
            return
    
    # Process data
    logger.info(f"Starting preprocessing from {scans_dir}")
    
    # Process each scene and save to the output directory
    for scene_id in tqdm(scenes_to_process, desc="Processing scenes"):
        scene_path = scans_dir / scene_id
        if not scene_path.exists():
            logger.warning(f"Scene path not found: {scene_path}")
            continue
        
        try:
            # Process the scene and save directly to output directory
            processor.process_scene(
                scene_path=str(scene_path),
                output_path=str(output_dir / f"{scene_id}.h5")
            )
        except Exception as e:
            logger.error(f"Error processing scene {scene_id}: {str(e)}")
            continue
    
    # Copy split files to output directory
    copy_split_files(output_dir)
    
    logger.info("Preprocessing completed successfully")

if __name__ == '__main__':
    main()