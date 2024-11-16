import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from pathlib import Path

class PointCloudVisualizer:
    """Visualization utilities for point cloud instance segmentation."""
    
    def __init__(self, config: dict):
        """Initialize visualizer with configuration."""
        self.config = config
        self.save_dir = Path(config['logging']['visualization']['save_dir'])
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
    def visualize_predictions(
        self,
        points: torch.Tensor,
        instance_labels: torch.Tensor,
        semantic_labels: Optional[torch.Tensor] = None,
        filename: Optional[str] = None
    ):
        """
        Visualize instance segmentation predictions.
        
        Args:
            points: (N, 3) tensor of point coordinates
            instance_labels: (N,) tensor of instance labels
            semantic_labels: (N,) tensor of semantic labels
            filename: Output filename
        """
        # Convert to numpy
        points = points.cpu().numpy()
        instance_labels = instance_labels.cpu().numpy()
        if semantic_labels is not None:
            semantic_labels = semantic_labels.cpu().numpy()
            
        # Create color map for instances
        unique_instances = np.unique(instance_labels)
        colors = self._get_instance_colors(len(unique_instances))
        
        # Assign colors to points
        point_colors = np.zeros((points.shape[0], 3))
        for i, inst_id in enumerate(unique_instances):
            mask = (instance_labels == inst_id)
            point_colors[mask] = colors[i]
            
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(point_colors)
        
        if filename is not None:
            # Save visualization
            o3d.io.write_point_cloud(
                str(self.save_dir / f"{filename}.ply"),
                pcd
            )
        else:
            # Interactive visualization
            o3d.visualization.draw_geometries([pcd])
            
    def plot_embeddings(
        self,
        embeddings: torch.Tensor,
        instance_labels: torch.Tensor,
        method: str = 'tsne',
        filename: Optional[str] = None
    ):
        """
        Visualize instance embeddings using dimensionality reduction.
        
        Args:
            embeddings: (N, D) tensor of instance embeddings
            instance_labels: (N,) tensor of instance labels
            method: Dimensionality reduction method ('tsne' or 'pca')
            filename: Output filename
        """
        # Convert to numpy
        embeddings = embeddings.cpu().numpy()
        instance_labels = instance_labels.cpu().numpy()
        
        # Perform dimensionality reduction
        if method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
        else:
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            
        reduced_embeddings = reducer.fit_transform(embeddings)
        
        # Create scatter plot
        unique_instances = np.unique(instance_labels)
        colors = self._get_instance_colors(len(unique_instances))
        
        plt.figure(figsize=(10, 10))
        for i, inst_id in enumerate(unique_instances):
            mask = (instance_labels == inst_id)
            plt.scatter(
                reduced_embeddings[mask, 0],
                reduced_embeddings[mask, 1],
                c=[colors[i]],
                label=f'Instance {inst_id}'
            )
            
        plt.title(f'Instance Embeddings ({method.upper()})')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        
        if filename is not None:
            plt.savefig(self.save_dir / f"{filename}.png")
        else:
            plt.show()
            
        plt.close()
        
    def visualize_results(
        self,
        metrics: Dict[str, List[float]],
        filename: Optional[str] = None
    ):
        """
        Plot training/validation metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            filename: Output filename
        """
        # Create multi-line plot
        fig = go.Figure()
        
        for metric_name, values in metrics.items():
            fig.add_trace(
                go.Scatter(
                    y=values,
                    mode='lines',
                    name=metric_name
                )
            )
            
        fig.update_layout(
            title='Training Metrics',
            xaxis_title='Epoch',
            yaxis_title='Value',
            hovermode='x'
        )
        
        if filename is not None:
            fig.write_html(str(self.save_dir / f"{filename}.html"))
        else:
            fig.show()
            
    def _get_instance_colors(self, num_instances: int) -> np.ndarray:
        """Generate distinct colors for instances."""
        colors = plt.cm.rainbow(np.linspace(0, 1, num_instances))[:, :3]
        return colors

    def create_comparison_visualization(
        self,
        points: torch.Tensor,
        pred_instances: torch.Tensor,
        gt_instances: torch.Tensor,
        filename: Optional[str] = None
    ):
        """
        Create side-by-side visualization of predicted and ground truth instances.
        
        Args:
            points: (N, 3) tensor of point coordinates
            pred_instances: (N,) tensor of predicted instance labels
            gt_instances: (N,) tensor of ground truth instance labels
            filename: Output filename
        """
        points = points.cpu().numpy()
        pred_instances = pred_instances.cpu().numpy()
        gt_instances = gt_instances.cpu().numpy()
        
        # Create colors for predictions and ground truth
        pred_colors = self._get_instance_colors(len(np.unique(pred_instances)))
        gt_colors = self._get_instance_colors(len(np.unique(gt_instances)))
        
        # Create point clouds
        pred_pcd = o3d.geometry.PointCloud()
        pred_pcd.points = o3d.utility.Vector3dVector(points)
        pred_pcd.colors = o3d.utility.Vector3dVector(
            pred_colors[pred_instances % len(pred_colors)]
        )
        
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(points)
        gt_pcd.colors = o3d.utility.Vector3dVector(
            gt_colors[gt_instances % len(gt_colors)]
        )
        
        # Visualize
        if filename is not None:
            o3d.io.write_point_cloud(
                str(self.save_dir / f"{filename}_pred.ply"),
                pred_pcd
            )
            o3d.io.write_point_cloud(
                str(self.save_dir / f"{filename}_gt.ply"),
                gt_pcd
            )
        else:
            o3d.visualization.draw_geometries([pred_pcd])
            o3d.visualization.draw_geometries([gt_pcd])