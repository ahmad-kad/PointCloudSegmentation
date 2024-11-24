import numpy as np
import h5py
from pathlib import Path
import open3d as o3d
import logging
from typing import Dict, Tuple, Optional

class DataProcessor:
    def __init__(self, config_path: str):
        """
        Initialize the DataProcessor for ScanNet data.
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger('DataProcessor')
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_point_cloud(self, ply_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load point cloud from PLY file.
        
        Returns:
            Tuple of (points, colors)
        """
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        return points, colors
    
    def _load_instance_labels(self, label_path: str) -> np.ndarray:
        """Load instance labels from file."""
        try:
            labels = np.load(label_path)
            return labels
        except Exception as e:
            self.logger.error(f"Error loading instance labels: {e}")
            return np.zeros(0, dtype=np.int32)
    
    def _normalize_points(self, points: np.ndarray) -> np.ndarray:
        """Normalize point coordinates to unit cube."""
        centroid = np.mean(points, axis=0)
        points = points - centroid
        max_distance = np.max(np.abs(points))
        points = points / max_distance
        return points
    
    def process_scene(self, scene_path: str, output_path: str):
        """
        Process a single ScanNet scene.
        
        Args:
            scene_path: Path to scene directory
            output_path: Path to save processed data
        """
        scene_path = Path(scene_path)
        
        # Load point cloud data
        points, colors = self._load_point_cloud(str(scene_path / f"{scene_path.name}_vh_clean_2.ply"))
        
        # Load instance labels
        instance_labels = self._load_instance_labels(str(scene_path / f"{scene_path.name}_vh_clean_2.instances.npy"))
        
        # Ensure we have labels for each point
        if len(instance_labels) != len(points):
            self.logger.error(f"Mismatch in points ({len(points)}) and labels ({len(instance_labels)})")
            return
        
        # Normalize point coordinates
        normalized_points = self._normalize_points(points)
        
        # Save processed data
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('points', data=normalized_points, dtype='float32')
            f.create_dataset('colors', data=colors, dtype='float32')
            f.create_dataset('instance_labels', data=instance_labels, dtype='int32')