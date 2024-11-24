import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import argparse
import open3d as o3d
from collections import defaultdict
import seaborn as sns

class EnhancedDataExplorer:
    def __init__(self, processed_dir: str):
        self.processed_dir = Path(processed_dir)
        self.logger = logging.getLogger('data_explorer')
        
        # Define ScanNet semantic label mapping
        self.label_mapping = {
            0: 'unannotated', 1: 'wall', 2: 'floor', 3: 'cabinet', 4: 'bed',
            5: 'chair', 6: 'sofa', 7: 'table', 8: 'door', 9: 'window',
            10: 'bookshelf', 11: 'picture', 12: 'counter', 13: 'desk',
            14: 'curtain', 15: 'refrigerator', 16: 'bathtub', 17: 'shower',
            18: 'toilet', 19: 'sink', 20: 'others'
        }
        
    def analyze_dataset(self):
        """Perform detailed analysis of the processed dataset."""
        files = list(self.processed_dir.glob('**/*.h5'))
        self.logger.info(f"Found {len(files)} processed files")
        
        stats = {
            'num_points': [],
            'num_instances': [],
            'semantic_label_counts': defaultdict(int),
            'instance_sizes': [],
            'color_stats': {'r': [], 'g': [], 'b': []},
            'point_bounds': {'x': [], 'y': [], 'z': []},
            'normals_available': 0,
            'features_available': defaultdict(int),
            'points_per_semantic_class': defaultdict(list),
            'instances_per_semantic_class': defaultdict(list)
        }
        
        for file_path in tqdm(files, desc="Analyzing files"):
            try:
                with h5py.File(file_path, 'r') as f:
                    # Basic counts
                    points = f['points'][:]
                    colors = f['colors'][:]
                    instance_labels = f['instance_labels'][:]
                    semantic_labels = f['semantic_labels'][:]
                    
                    stats['num_points'].append(len(points))
                    unique_instances = np.unique(instance_labels)
                    stats['num_instances'].append(len(unique_instances))
                    
                    # Semantic label distribution
                    unique_labels, label_counts = np.unique(semantic_labels, return_counts=True)
                    for label, count in zip(unique_labels, label_counts):
                        stats['semantic_label_counts'][int(label)] += count
                        
                    # Instance sizes
                    for instance_id in unique_instances:
                        if instance_id > 0:  # Skip background
                            instance_size = np.sum(instance_labels == instance_id)
                            stats['instance_sizes'].append(instance_size)
                    
                    # Color statistics
                    stats['color_stats']['r'].extend(colors[:, 0])
                    stats['color_stats']['g'].extend(colors[:, 1])
                    stats['color_stats']['b'].extend(colors[:, 2])
                    
                    # Point cloud bounds
                    stats['point_bounds']['x'].extend(points[:, 0])
                    stats['point_bounds']['y'].extend(points[:, 1])
                    stats['point_bounds']['z'].extend(points[:, 2])
                    
                    # Feature availability
                    if 'features' in f:
                        for feature_name in f['features'].keys():
                            stats['features_available'][feature_name] += 1
                        if 'normals' in f['features']:
                            stats['normals_available'] += 1
                    
                    # Points per semantic class
                    for label in unique_labels:
                        points_in_class = np.sum(semantic_labels == label)
                        stats['points_per_semantic_class'][int(label)].append(points_in_class)
                    
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {str(e)}")
        
        return stats
    
    def visualize_statistics(self, stats, output_dir: str = 'analysis'):
        """Create visualizations of dataset statistics."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 1. Semantic Label Distribution
        plt.figure(figsize=(15, 6))
        labels = []
        counts = []
        for label_id, count in stats['semantic_label_counts'].items():
            labels.append(self.label_mapping.get(label_id, f'unknown_{label_id}'))
            counts.append(count)
        
        plt.bar(labels, counts)
        plt.xticks(rotation=45, ha='right')
        plt.title('Distribution of Semantic Labels')
        plt.ylabel('Number of Points')
        plt.tight_layout()
        plt.savefig(output_dir / 'semantic_distribution.png')
        plt.close()
        
        # 2. Instance Size Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(stats['instance_sizes'], bins=50)
        plt.title('Instance Size Distribution')
        plt.xlabel('Number of Points per Instance')
        plt.ylabel('Frequency')
        plt.savefig(output_dir / 'instance_sizes.png')
        plt.close()
        
        # 3. Point Cloud Bounds
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, (coord, values) in zip(axes, stats['point_bounds'].items()):
            ax.hist(values, bins=50)
            ax.set_title(f'{coord} Coordinate Distribution')
        plt.tight_layout()
        plt.savefig(output_dir / 'spatial_distribution.png')
        plt.close()
        
        # 4. Color Distribution
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, (color, values) in zip(axes, stats['color_stats'].items()):
            ax.hist(values, bins=50)
            ax.set_title(f'{color} Channel Distribution')
        plt.tight_layout()
        plt.savefig(output_dir / 'color_distribution.png')
        plt.close()
        
        # Save numerical statistics
        with open(output_dir / 'dataset_statistics.txt', 'w') as f:
            f.write("Dataset Statistics Summary\n")
            f.write("========================\n\n")
            
            f.write("General Statistics:\n")
            f.write(f"Total number of scenes: {len(stats['num_points'])}\n")
            f.write(f"Average points per scene: {np.mean(stats['num_points']):.1f}\n")
            f.write(f"Average instances per scene: {np.mean(stats['num_instances']):.1f}\n")
            f.write(f"Scenes with normals: {stats['normals_available']}\n\n")
            
            f.write("Available Features:\n")
            for feature, count in stats['features_available'].items():
                f.write(f"{feature}: {count} scenes\n")
            
            f.write("\nSemantic Label Statistics:\n")
            for label_id, count in stats['semantic_label_counts'].items():
                label_name = self.label_mapping.get(label_id, f'unknown_{label_id}')
                f.write(f"{label_name}: {count} points\n")
            
def main():
    parser = argparse.ArgumentParser(description='Enhanced data exploration')
    parser.add_argument('--processed-dir', type=str, required=True, help='Directory with processed data')
    parser.add_argument('--output-dir', type=str, default='analysis', help='Output directory for analysis')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    explorer = EnhancedDataExplorer(args.processed_dir)
    
    # Run analysis
    logging.info("Starting detailed dataset analysis...")
    stats = explorer.analyze_dataset()
    
    # Create visualizations
    logging.info("Generating visualizations...")
    explorer.visualize_statistics(stats, args.output_dir)
    
    logging.info(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()