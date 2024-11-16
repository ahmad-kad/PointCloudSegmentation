import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import h5py
import yaml
from tqdm import tqdm

class DataProcessor:
    """Utility class for data processing and management."""
    
    def __init__(self, config_path: str):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def preprocess_dataset(self,
                          raw_data_dir: str,
                          output_dir: str,
                          split: str = 'train'):
        """
        Preprocess raw point cloud data.
        
        Args:
            raw_data_dir: Directory containing raw data
            output_dir: Output directory for processed data
            split: Data split (train/val/test)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Get file list
        raw_data_dir = Path(raw_data_dir)
        files = list(raw_data_dir.glob('*.ply'))
        
        for file_path in tqdm(files, desc=f"Processing {split} data"):
            # Load point cloud
            points, colors, instances = self._load_raw_data(file_path)
            
            # Compute features
            features = self._compute_features(points, colors)
            
            # Subsample if needed
            if self.config['dataset']['num_points'] > 0:
                points, colors, instances, features = self._subsample(
                    points, colors, instances, features,
                    self.config['dataset']['num_points']
                )
                
            # Save processed data
            output_path = output_dir / f"{file_path.stem}.h5"
            self._save_processed_data(
                output_path,
                points, colors, instances, features
            )
            
    def _load_raw_data(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load raw point cloud data."""
        import open3d as o3d
        
        # Load PLY file
        pcd = o3d.io.read_point_cloud(str(file_path))
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        # Load instance labels (assuming they're in a separate file)
        instance_path = file_path.parent / 'instances' / f"{file_path.stem}.txt"
        if instance_path.exists():
            instances = np.loadtxt(instance_path, dtype=np.int32)
        else:
            instances = np.zeros(len(points), dtype=np.int32)
            
        return points, colors, instances
    
    def _compute_features(self,
                         points: np.ndarray,
                         colors: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute point features."""
        features = {}
        
        if self.config['features']['geometric']['enable']:
            # Compute geometric features
            features['normals'] = self._compute_normals(points)
            features['curvature'] = self._compute_curvature(points, features['normals'])
            
            if self.config['features']['geometric']['eigenfeatures']['enable']:
                eigen_features = self._compute_eigenfeatures(
                    points,
                    k=self.config['features']['geometric']['eigenfeatures']['k_neighbors']
                )
                features.update(eigen_features)
                
        if self.config['features']['contextual']['enable']:
            # Compute contextual features
            features['density'] = self._compute_density(
                points,
                radius=self.config['features']['contextual']['density']['radius']
            )
            
            if self.config['features']['contextual']['knn']['enable']:
                knn_features = self._compute_knn_features(
                    points, colors,
                    k=self.config['features']['contextual']['knn']['k_neighbors']
                )
                features.update(knn_features)
                
        return features
    
    def _compute_normals(self, points: np.ndarray) -> np.ndarray:
        """Compute point normals."""
        import open3d as o3d
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals()
        
        return np.asarray(pcd.normals)
    
    def _compute_curvature(self,
                          points: np.ndarray,
                          normals: np.ndarray) -> np.ndarray:
        """Compute point curvature."""
        from sklearn.neighbors import NearestNeighbors
        
        # Find k nearest neighbors
        k = self.config['features']['geometric']['curvature']['k_neighbors']
        nbrs = NearestNeighbors(n_neighbors=k).fit(points)
        distances, indices = nbrs.kneighbors(points)
        
        # Compute curvature as variance of normals
        curvature = np.zeros(len(points))
        for i in range(len(points)):
            neighbor_normals = normals[indices[i]]
            curvature[i] = 1 - np.abs(np.mean(neighbor_normals.dot(normals[i])))
            
        return curvature
    
    def _compute_eigenfeatures(self,
                             points: np.ndarray,
                             k: int) -> Dict[str, np.ndarray]:
        """Compute eigenvalue-based features."""
        from sklearn.neighbors import NearestNeighbors
        
        nbrs = NearestNeighbors(n_neighbors=k).fit(points)
        distances, indices = nbrs.kneighbors(points)
        
        features = {}
        for i in range(len(points)):
            # Compute covariance matrix
            neighbors = points[indices[i]]
            centered = neighbors - neighbors.mean(axis=0)
            cov = centered.T @ centered
            
            # Compute eigenvalues
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]
            
            # Compute features
            if 'linearity' in self.config['features']['geometric']['eigenfeatures']['features']:
                features.setdefault('linearity', []).append(
                    (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]
                )
                
            if 'planarity' in self.config['features']['geometric']['eigenfeatures']['features']:
                features.setdefault('planarity', []).append(
                    (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0]
                )
                
            if 'sphericity' in self.config['features']['geometric']['eigenfeatures']['features']:
                features.setdefault('sphericity', []).append(
                    eigenvalues[2] / eigenvalues[0]
                )
                
        return {k: np.array(v) for k, v in features.items()}
    
    def _compute_density(self,
                        points: np.ndarray,
                        radius: float) -> np.ndarray:
        """Compute local point density."""
        from sklearn.neighbors import NearestNeighbors
        
        nbrs = NearestNeighbors(radius=radius).fit(points)
        distances, indices = nbrs.radius_neighbors(points)
        
        return np.array([len(idx) for idx in indices])
    
    def _compute_knn_features(self,
                            points: np.ndarray,
                            colors: np.ndarray,
                            k: int) -> Dict[str, np.ndarray]:
        """Compute k-nearest neighbor features."""
        from sklearn.neighbors import NearestNeighbors
        
        nbrs = NearestNeighbors(n_neighbors=k).fit(points)
        distances, indices = nbrs.kneighbors(points)
        
        features = {}
        if 'mean' in self.config['features']['contextual']['knn']['features']:
            features['color_mean'] = np.array([
                colors[idx].mean(axis=0) for idx in indices
            ])
            
        if 'std' in self.config['features']['contextual']['knn']['features']:
            features['color_std'] = np.array([
                colors[idx].std(axis=0) for idx in indices
            ])
            
        if 'max' in self.config['features']['contextual']['knn']['features']:
            features['height_max'] = np.array([
                points[idx][:, 2].max() for idx in indices
            ])
            
        return features
    
    def _subsample(self,
                  points: np.ndarray,
                  colors: np.ndarray,
                  instances: np.ndarray,
                  features: Dict[str, np.ndarray],
                  num_points: int) -> Tuple:
        """Randomly subsample points."""
        if len(points) <= num_points:
            return points, colors, instances, features
            
        indices = np.random.choice(
            len(points), num_points, replace=False
        )
        
        subsampled_features = {
            k: v[indices] for k, v in features.items()
        }
        
        return (
            points[indices],
            colors[indices],
            instances[indices],
            subsampled_features
        )
    
    def _save_processed_data(self,
                           output_path: Path,
                           points: np.ndarray,
                           colors: np.ndarray,
                           instances: np.ndarray,
                           features: Dict[str, np.ndarray]):
        """Save processed data to HDF5 format."""
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('points', data=points)
            f.create_dataset('colors', data=colors)
            f.create_dataset('instances', data=instances)
            
            features_group = f.create_group('features')
            for name, feature in features.items():
                features_group.create_dataset(name, data=feature)