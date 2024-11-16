import torch
import numpy as np
from typing import Tuple, Optional
from scipy.spatial import KDTree

def estimate_normals(
    points: torch.Tensor,
    k: int = 30,
    orientation_reference: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Estimate point normals using PCA.
    
    Args:
        points: (N, 3) tensor of point coordinates
        k: Number of neighbors for normal estimation
        orientation_reference: Optional reference point for normal orientation
        
    Returns:
        normals: (N, 3) tensor of point normals
    """
    points_np = points.cpu().numpy()
    tree = KDTree(points_np)
    
    # Query k nearest neighbors for each point
    distances, indices = tree.query(points_np, k=k)
    
    # Compute normals using PCA
    normals = np.zeros_like(points_np)
    for i in range(len(points_np)):
        neighbors = points_np[indices[i]]
        centered = neighbors - neighbors.mean(axis=0)
        cov = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        normal = eigenvectors[:, 0]  # Smallest eigenvector
        
        # Orient normal
        if orientation_reference is not None:
            reference = orientation_reference[i].cpu().numpy()
            if normal.dot(reference - points_np[i]) < 0:
                normal = -normal
                
        normals[i] = normal
        
    return torch.from_numpy(normals).to(points.device)

def compute_fpfh_features(
    points: torch.Tensor,
    normals: torch.Tensor,
    radius: float = 0.1
) -> torch.Tensor:
    """
    Compute Fast Point Feature Histograms (FPFH).
    
    Args:
        points: (N, 3) tensor of point coordinates
        normals: (N, 3) tensor of point normals
        radius: Radius for neighbor search
        
    Returns:
        fpfh: (N, 33) tensor of FPFH features
    """
    import open3d as o3d
    
    # Convert to Open3D format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    pcd.normals = o3d.utility.Vector3dVector(normals.cpu().numpy())
    
    # Compute FPFH features
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamRadius(radius)
    )
    
    return torch.from_numpy(np.asarray(fpfh.data).T).float().to(points.device)

def voxelize_points(
    points: torch.Tensor,
    features: Optional[torch.Tensor],
    voxel_size: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Voxelize point cloud.
    
    Args:
        points: (N, 3) tensor of point coordinates
        features: (N, C) tensor of point features
        voxel_size: Voxel size
        
    Returns:
        voxel_coords: (M, 3) tensor of voxel coordinates
        voxel_features: (M, C) tensor of voxel features
        voxel_indices: (N,) tensor mapping original points to voxels
    """
    # Quantize coordinates
    coords = torch.floor(points / voxel_size)
    coords, indices, counts = torch.unique(
        coords,
        dim=0,
        return_inverse=True,
        return_counts=True
    )
    
    if features is not None:
        # Average features in each voxel
        voxel_features = torch.zeros(
            (len(coords), features.shape[1]),
            device=points.device,
            dtype=features.dtype
        )
        for i in range(features.shape[1]):
            voxel_features[:, i] = torch.bincount(
                indices,
                weights=features[:, i],
                minlength=len(coords)
            ) / counts.float()
    else:
        voxel_features = None
        
    return coords * voxel_size, voxel_features, indices

def compute_local_reference_frame(
    points: torch.Tensor,
    k: int = 30
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute local reference frame for each point.
    
    Args:
        points: (N, 3) tensor of point coordinates
        k: Number of neighbors
        
    Returns:
        x_axis: (N, 3) local x-axis
        y_axis: (N, 3) local y-axis
        z_axis: (N, 3) local z-axis
    """
    normals = estimate_normals(points, k)  # z-axis
    
    # Compute x-axis (arbitrary vector perpendicular to normal)
    x_axis = torch.zeros_like(points)
    for i in range(len(points)):
        normal = normals[i]
        # Find least aligned global axis
        dots = torch.abs(torch.tensor([
            normal.dot(torch.tensor([1., 0., 0.]).to(points.device)),
            normal.dot(torch.tensor([0., 1., 0.]).to(points.device)),
            normal.dot(torch.tensor([0., 0., 1.]).to(points.device))
        ]))
        least_aligned = torch.argmin(dots)
        
        if least_aligned == 0:
            v = torch.tensor([1., 0., 0.])
        elif least_aligned == 1:
            v = torch.tensor([0., 1., 0.])
        else:
            v = torch.tensor([0., 0., 1.])
            
        v = v.to(points.device)
        x_axis[i] = torch.cross(normal, v)
        x_axis[i] = x_axis[i] / torch.norm(x_axis[i])
        
    # Compute y-axis using cross product
    y_axis = torch.cross(normals, x_axis)
    
    return x_axis, y_axis, normals