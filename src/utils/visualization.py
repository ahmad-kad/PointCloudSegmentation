import torch
import numpy as np
import open3d as o3d
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random

class SegmentationVisualizer:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate random colors for instances
        random.seed(42)
        self.color_map = {}
    
    def get_instance_color(self, instance_id):
        """Get consistent color for an instance ID."""
        if instance_id not in self.color_map:
            self.color_map[instance_id] = np.array([
                random.random(),
                random.random(),
                random.random()
            ])
        return self.color_map[instance_id]
    
    def visualize_point_cloud(
        self,
        points: np.ndarray,
        instance_labels: np.ndarray,
        filename: str,
        original_colors: np.ndarray = None
    ):
        """
        Visualize point cloud with instance segmentation.
        
        Args:
            points: (N, 3) point coordinates
            instance_labels: (N,) instance labels
            filename: Output filename
            original_colors: Optional (N, 3) original point colors
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Color points by instance
        colors = np.zeros_like(points)
        for instance_id in np.unique(instance_labels):
            mask = instance_labels == instance_id
            colors[mask] = self.get_instance_color(instance_id)
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Save visualization
        o3d.io.write_point_cloud(
            str(self.output_dir / f"{filename}_instances.ply"),
            pcd
        )
        
        # Save original colors if provided
        if original_colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(original_colors)
            o3d.io.write_point_cloud(
                str(self.output_dir / f"{filename}_original.ply"),
                pcd
            )
    
    def visualize_embeddings(
        self,
        embeddings: np.ndarray,
        instance_labels: np.ndarray,
        filename: str
    ):
        """
        Visualize embeddings using t-SNE.
        
        Args:
            embeddings: (N, D) point embeddings
            instance_labels: (N,) instance labels
            filename: Output filename
        """
        # Reduce dimensionality for visualization
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Plot
        plt.figure(figsize=(10, 10))
        unique_instances = np.unique(instance_labels)
        
        for instance_id in unique_instances:
            mask = instance_labels == instance_id
            color = self.get_instance_color(instance_id)
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[color],
                label=f'Instance {instance_id}',
                alpha=0.6
            )
        
        plt.title('t-SNE Visualization of Point Embeddings')
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{filename}_embeddings.png")
        plt.close()
    
    def save_metrics_plot(
        self,
        train_losses: list,
        val_losses: list,
        filename: str = 'training_curves'
    ):
        """Plot training and validation losses."""
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{filename}.png")
        plt.close()