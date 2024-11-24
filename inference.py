import torch
import numpy as np
import open3d as o3d
import argparse
from pathlib import Path
import yaml
import logging
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import torch.nn.functional as F

from src.models.point2net import Point2Net

class HighPrecisionLabeler:
    @staticmethod
    def encode_label_to_color(label: int) -> np.ndarray:
        """Encode a 24-bit label into RGB values."""
        r = (label & 0xFF0000) >> 16
        g = (label & 0x00FF00) >> 8
        b = label & 0x0000FF
        return np.array([r, g, b]) / 255.0

    @staticmethod
    def decode_color_to_label(color: np.ndarray) -> int:
        """Decode RGB values back to label."""
        color = np.round(color * 255).astype(int)
        return (color[0] << 16) | (color[1] << 8) | color[2]

    def labels_to_colors(self, labels: np.ndarray) -> np.ndarray:
        """Convert array of labels to colors."""
        colors = np.zeros((len(labels), 3))
        for i, label in enumerate(labels):
            colors[i] = self.encode_label_to_color(label)
        return colors

    def colors_to_labels(self, colors: np.ndarray) -> np.ndarray:
        """Convert array of colors back to labels."""
        labels = np.zeros(len(colors), dtype=np.int32)
        for i, color in enumerate(colors):
            labels[i] = self.decode_color_to_label(color)
        return labels

class PointCloudInference:
    def __init__(self, checkpoint_path: str, config_path: str, device: str = 'cuda'):
        """
        Initialize inference setup.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            config_path: Path to model configuration file
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.config = self.load_config(config_path)
        self.model = self.load_model(checkpoint_path)
        self.labeler = HighPrecisionLabeler()
        
    def load_config(self, config_path: str) -> dict:
        """Load model configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_model(self, checkpoint_path: str) -> torch.nn.Module:
        """Load trained model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model = Point2Net(
            feature_dim=6,  # 3 for coordinates, 3 for colors
            embedding_dim=self.config['model']['embedding_dim']
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    def preprocess_point_cloud(self, points: np.ndarray, colors: np.ndarray = None) -> torch.Tensor:
        """
        Preprocess point cloud for inference.
        
        Args:
            points: (N, 3) point coordinates
            colors: (N, 3) point colors, normalized to [0, 1]
        
        Returns:
            (1, N, 6) tensor of concatenated coordinates and colors
        """
        # Normalize coordinates
        centroid = np.mean(points, axis=0)
        points = points - centroid
        scale = np.max(np.abs(points))
        points = points / scale
        
        # Handle colors
        if colors is None:
            colors = np.ones_like(points) * 0.5
            
        # Ensure colors are normalized
        if colors.max() > 1:
            colors = colors / 255.0
        
        # Combine features
        features = np.concatenate([points, colors], axis=1)
        features = torch.from_numpy(features).float().unsqueeze(0)
        return features, centroid, scale
    
    def cluster_embeddings(
        self,
        embeddings: np.ndarray,
        eps: float = 0.1,
        min_samples: int = 50
    ) -> np.ndarray:
        """
        Cluster embeddings using DBSCAN.
        
        Args:
            embeddings: (N, D) embedding vectors
            eps: DBSCAN epsilon parameter
            min_samples: DBSCAN min_samples parameter
        
        Returns:
            (N,) array of instance labels
        """
        # Normalize embeddings
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Perform clustering
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        instance_labels = clusterer.fit_predict(embeddings)
        
        # Relabel instances from 1 (reserve 0 for noise points)
        unique_labels = np.unique(instance_labels)
        new_labels = np.zeros_like(instance_labels)
        
        label_counter = 1  # Start from 1 to reserve 0 for noise
        for label in unique_labels:
            if label == -1:
                new_labels[instance_labels == label] = 0  # Noise points get label 0
            else:
                new_labels[instance_labels == label] = label_counter
                label_counter += 1
        
        return new_labels
    
    @torch.no_grad()
    def predict(
        self,
        points: np.ndarray,
        colors: np.ndarray = None,
        eps: float = 0.1,
        min_samples: int = 50
    ) -> dict:
        """
        Perform instance segmentation on a point cloud.
        
        Args:
            points: (N, 3) point coordinates
            colors: (N, 3) point colors
            eps: DBSCAN epsilon parameter
            min_samples: DBSCAN minimum points parameter
        
        Returns:
            Dictionary containing:
                - instance_labels: (N,) instance labels
                - embeddings: (N, D) point embeddings
                - num_instances: Number of instances found
        """
        # Preprocess point cloud
        features, centroid, scale = self.preprocess_point_cloud(points, colors)
        features = features.to(self.device)
        
        # Get embeddings
        embeddings = self.model(features)
        embeddings = embeddings.cpu().numpy().squeeze()
        
        # Cluster embeddings
        instance_labels = self.cluster_embeddings(
            embeddings,
            eps=eps,
            min_samples=min_samples
        )
        
        return {
            'instance_labels': instance_labels,
            'embeddings': embeddings,
            'num_instances': len(np.unique(instance_labels)) - 1,  # Subtract 1 to exclude noise label
            'centroid': centroid,
            'scale': scale
        }
    
    def save_results(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        instance_labels: np.ndarray,
        output_path: str
    ):
        """Save results in multiple formats."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save labeled point cloud with encoded instance labels
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Save instance labels encoded as colors
        instance_colors = self.labeler.labels_to_colors(instance_labels)
        pcd.colors = o3d.utility.Vector3dVector(instance_colors)
        o3d.io.write_point_cloud(str(output_path), pcd)
        
        # Save visualization with original colors
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(
                str(output_path.parent / f"{output_path.stem}_original.ply"),
                pcd
            )
        
        # Save raw labels as numpy array for perfect recovery
        np.save(str(output_path.parent / f"{output_path.stem}_labels.npy"), instance_labels)

def main():
    parser = argparse.ArgumentParser(description='Point cloud instance segmentation inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to model configuration')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input point cloud (PLY format)')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save results')
    parser.add_argument('--eps', type=float, default=0.1,
                       help='DBSCAN epsilon parameter')
    parser.add_argument('--min-samples', type=int, default=50,
                       help='DBSCAN min_samples parameter')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run inference on')
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('inference')
    
    # Initialize inference
    logger.info("Initializing model...")
    inferencer = PointCloudInference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    # Load point cloud
    logger.info(f"Loading point cloud from {args.input}")
    pcd = o3d.io.read_point_cloud(args.input)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # Run inference
    logger.info("Running inference...")
    results = inferencer.predict(
        points,
        colors,
        eps=args.eps,
        min_samples=args.min_samples
    )
    
    # Save results
    logger.info(f"Found {results['num_instances']} instances")
    logger.info(f"Saving results to {args.output}")
    inferencer.save_results(
        points,
        colors,
        results['instance_labels'],
        args.output
    )
    
    logger.info("Done!")

if __name__ == '__main__':
    main()