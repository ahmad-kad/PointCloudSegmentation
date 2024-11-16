import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from .blocks import SAModule, FPModule

class PointNet2InstanceSegmentation(nn.Module):
    """PointNet++ architecture for instance segmentation."""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        # Set up set abstraction layers
        self.sa_modules = nn.ModuleList()
        for i in range(len(config['backbone']['npoints'])):
            sa_module = SAModule(
                npoint=config['backbone']['npoints'][i],
                radius=config['backbone']['radius'][i],
                nsample=config['backbone']['nsamples'][i],
                in_channel=3 if i == 0 else config['backbone']['mlps'][i-1][-1],
                mlp=config['backbone']['mlps'][i],
                use_xyz=config['backbone']['use_xyz']
            )
            self.sa_modules.append(sa_module)
            
        # Set up feature propagation layers
        self.fp_modules = nn.ModuleList()
        for i in range(len(config['fp_layers']['mlps'])):
            fp_module = FPModule(
                in_channel=config['backbone']['mlps'][-i-1][-1] + \
                          (config['backbone']['mlps'][-i-2][-1] if i < len(config['fp_layers']['mlps'])-1 else 0),
                mlp=config['fp_layers']['mlps'][i]
            )
            self.fp_modules.append(fp_module)
            
        # Instance segmentation head
        self.instance_head = InstanceHead(
            in_channels=config['fp_layers']['mlps'][-1][-1],
            embedding_dim=config['instance_head']['embedding_dim'],
            hidden_dims=config['instance_head']['hidden_dims'],
            num_classes=config['instance_head']['num_classes'],
            dropout=config['instance_head']['dropout']
        )
        
    def forward(self, 
                points: torch.Tensor, 
                features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the network.
        
        Args:
            points: (B, N, 3) tensor of point coordinates
            features: (B, N, C) tensor of point features
            
        Returns:
            Dict containing:
                - semantic_logits: (B, N, num_classes) semantic segmentation logits
                - instance_embeddings: (B, N, embedding_dim) instance embeddings
                - center_predictions: (B, N, 3) center predictions
        """
        xyz = points.contiguous()
        features = features.transpose(1, 2).contiguous() if features is not None else None
        
        # Save intermediate features for skip connections
        xyz_list = [xyz]
        features_list = [features]
        
        # Set abstraction layers
        for i in range(len(self.sa_modules)):
            xyz, features = self.sa_modules[i](xyz, features)
            xyz_list.append(xyz)
            features_list.append(features)
            
        # Feature propagation layers
        for i in range(len(self.fp_modules)):
            features = self.fp_modules[i](
                xyz_list[-i-2],
                xyz_list[-i-1],
                features_list[-i-2],
                features_list[-i-1]
            )
            
        # Instance segmentation head
        results = self.instance_head(features, xyz)
        
        return results

class InstanceHead(nn.Module):
    """Instance segmentation head."""
    
    def __init__(self,
                 in_channels: int,
                 embedding_dim: int,
                 hidden_dims: List[int],
                 num_classes: int,
                 dropout: float = 0.3):
        super().__init__()
        
        # Semantic segmentation branch
        semantic_layers = []
        curr_dim = in_channels
        for hidden_dim in hidden_dims:
            semantic_layers.extend([
                nn.Conv1d(curr_dim, hidden_dim, 1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            curr_dim = hidden_dim
        semantic_layers.append(nn.Conv1d(curr_dim, num_classes, 1))
        self.semantic_branch = nn.Sequential(*semantic_layers)
        
        # Instance embedding branch
        embedding_layers = []
        curr_dim = in_channels
        for hidden_dim in hidden_dims:
            embedding_layers.extend([
                nn.Conv1d(curr_dim, hidden_dim, 1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            curr_dim = hidden_dim
        embedding_layers.append(nn.Conv1d(curr_dim, embedding_dim, 1))
        self.embedding_branch = nn.Sequential(*embedding_layers)
        
        # Center prediction branch
        center_layers = []
        curr_dim = in_channels
        for hidden_dim in hidden_dims:
            center_layers.extend([
                nn.Conv1d(curr_dim, hidden_dim, 1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            curr_dim = hidden_dim
        center_layers.append(nn.Conv1d(curr_dim, 3, 1))
        self.center_branch = nn.Sequential(*center_layers)
        
    def forward(self, 
                features: torch.Tensor, 
                xyz: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of instance segmentation head.
        
        Args:
            features: (B, C, N) tensor of point features
            xyz: (B, N, 3) tensor of point coordinates
            
        Returns:
            Dict containing semantic logits, instance embeddings, and center predictions
        """
        # Semantic segmentation
        semantic_logits = self.semantic_branch(features)
        
        # Instance embeddings
        instance_embeddings = self.embedding_branch(features)
        instance_embeddings = F.normalize(instance_embeddings, p=2, dim=1)
        
        # Center predictions
        center_predictions = self.center_branch(features)
        
        return {
            'semantic_logits': semantic_logits.transpose(1, 2),
            'instance_embeddings': instance_embeddings.transpose(1, 2),
            'center_predictions': center_predictions.transpose(1, 2)
        }