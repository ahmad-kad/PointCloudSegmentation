import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from typing import List, Optional, Tuple

class SAModule(nn.Module):
    """Set Abstraction Module for PointNet++."""
    
    def __init__(
        self,
        npoint: int,
        radius: float,
        nsample: int,
        in_channel: int,
        mlp: List[int],
        use_xyz: bool = True
    ):
        super().__init__()
        
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.use_xyz = use_xyz
        
        # Build MLPs
        if use_xyz:
            in_channel += 3
            
        self.mlp = nn.ModuleList()
        for i in range(len(mlp)):
            if i == 0:
                linear = nn.Conv2d(in_channel, mlp[i], 1)
            else:
                linear = nn.Conv2d(mlp[i-1], mlp[i], 1)
            self.mlp.append(linear)
            self.mlp.append(nn.BatchNorm2d(mlp[i]))
            self.mlp.append(nn.ReLU(inplace=True))
            
    def forward(self,
                xyz: torch.Tensor,
                features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            xyz: (B, N, 3) tensor of point coordinates
            features: (B, C, N) tensor of point features
            
        Returns:
            new_xyz: (B, npoint, 3) subsampled points
            new_features: (B, C', npoint) subsampled features
        """
        # Sample points
        fps_idx = self.farthest_point_sample(xyz, self.npoint)
        new_xyz = self.index_points(xyz, fps_idx)
        
        # Group points
        idx = self.ball_query(xyz, new_xyz, self.radius, self.nsample)
        grouped_xyz = self.index_points(xyz, idx)
        grouped_xyz -= new_xyz.unsqueeze(2)
        
        if features is not None:
            grouped_features = self.index_points(
                features.transpose(1, 2),
                idx
            )
            if self.use_xyz:
                grouped_features = torch.cat(
                    [grouped_xyz, grouped_features],
                    dim=-1
                )
        else:
            grouped_features = grouped_xyz
            
        # Apply MLPs
        grouped_features = grouped_features.permute(0, 3, 2, 1)
        for layer in self.mlp:
            grouped_features = layer(grouped_features)
            
        # Max pooling
        new_features = torch.max(grouped_features, dim=2)[0]
        
        return new_xyz, new_features
    
    @staticmethod
    def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """Farthest point sampling."""
        device = xyz.device
        B, N, _ = xyz.shape
        
        centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
        distance = torch.ones(B, N, device=device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
        batch_indices = torch.arange(B, dtype=torch.long, device=device)
        
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
            
        return centroids
    
    @staticmethod
    def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Index points by given indices."""
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points
    
    @staticmethod
    def ball_query(xyz: torch.Tensor, 
                   new_xyz: torch.Tensor,
                   radius: float,
                   nsample: int) -> torch.Tensor:
        """Query ball points."""
        device = xyz.device
        B, N, _ = xyz.shape
        _, S, _ = new_xyz.shape
        
        group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
        sqrdists = square_distance(new_xyz, xyz)
        group_idx[sqrdists > radius ** 2] = N
        group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
        mask = group_idx == N
        group_idx[mask] = group_first[mask]
        return group_idx

class FPModule(nn.Module):
    """Feature Propagation Module for PointNet++."""
    
    def __init__(self, in_channel: int, mlp: List[int]):
        super().__init__()
        
        self.mlp = nn.ModuleList()
        for i in range(len(mlp)):
            if i == 0:
                linear = nn.Conv1d(in_channel, mlp[i], 1)
            else:
                linear = nn.Conv1d(mlp[i-1], mlp[i], 1)
            self.mlp.append(linear)
            self.mlp.append(nn.BatchNorm1d(mlp[i]))
            self.mlp.append(nn.ReLU(inplace=True))
            
    def forward(self,
                xyz1: torch.Tensor,
                xyz2: torch.Tensor,
                features1: Optional[torch.Tensor],
                features2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of feature propagation.
        
        Args:
            xyz1: (B, N, 3) coordinates of points to propagate to
            xyz2: (B, M, 3) coordinates of points to propagate from
            features1: (B, C1, N) features of xyz1
            features2: (B, C2, M) features of xyz2
            
        Returns:
            new_features: (B, C', N) propagated features
        """
        if features1 is not None:
            features1 = features1.permute(0, 2, 1)
        
        dists = square_distance(xyz1, xyz2)
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :3], idx[:, :, :3]
        
        # Calculate weights
        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        
        # Interpolate features
        interpolated_feats = torch.zeros_like(features1)
        for i in range(3):
            interpolated_feats += weight[:, :, i:i+1] * \
                                features2.permute(0, 2, 1)[:, torch.arange(xyz1.shape[1]).unsqueeze(1), idx[:, :, i]]
                                
        if features1 is not None:
            new_features = torch.cat([interpolated_feats, features1], dim=-1)
        else:
            new_features = interpolated_feats
            
        # Apply MLPs
        new_features = new_features.permute(0, 2, 1)
        for layer in self.mlp:
            new_features = layer(new_features)
            
        return new_features

class BasicBlock(nn.Module):
    """Basic residual block for Sparse CNN."""
    
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 dimension: int = 3):
        super().__init__()
        
        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=3,
            stride=stride, dimension=dimension
        )
        self.bn1 = ME.MinkowskiBatchNorm(planes)
        self.relu = ME.MinkowskiReLU(inplace=True)
        
        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3,
            stride=1, dimension=dimension
        )
        self.bn2 = ME.MinkowskiBatchNorm(planes)
        
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    inplanes, planes, kernel_size=1,
                    stride=stride, dimension=dimension
                ),
                ME.MinkowskiBatchNorm(planes)
            )
        else:
            self.downsample = None
            
    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        """Forward pass."""
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Calculate squared distance between each two points."""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist