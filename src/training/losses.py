import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class InstanceSegLoss(nn.Module):
    """Combined loss for instance segmentation."""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        # Initialize semantic segmentation loss
        self.semantic_loss = nn.CrossEntropyLoss()
        
        # Initialize center prediction loss
        self.center_loss = CenterLoss(config['loss']['center_loss'])
        
        # Initialize discriminative loss for instance embeddings
        self.discriminative_loss = DiscriminativeLoss(
            config['loss']['discriminative_loss']
        )
        
        # Loss weights
        self.semantic_weight = config['loss']['semantic_weight']
        self.center_weight = config['loss']['center_weight']
        self.instance_weight = config['loss']['instance_weight']
        
    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined loss.
        
        Args:
            predictions: Dict containing model predictions
            targets: Dict containing ground truth
            
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary of individual losses
        """
        # Semantic segmentation loss
        semantic_loss = self.semantic_loss(
            predictions['semantic_logits'],
            targets['semantic_labels']
        )
        
        # Center prediction loss
        center_loss = self.center_loss(
            predictions['center_predictions'],
            targets['center_labels']
        )
        
        # Instance embedding loss
        instance_loss = self.discriminative_loss(
            predictions['instance_embeddings'],
            targets['instance_labels']
        )
        
        # Combine losses
        total_loss = (
            self.semantic_weight * semantic_loss +
            self.center_weight * center_loss +
            self.instance_weight * instance_loss
        )
        
        # Create loss dictionary
        loss_dict = {
            'total_loss': total_loss.item(),
            'semantic_loss': semantic_loss.item(),
            'center_loss': center_loss.item(),
            'instance_loss': instance_loss.item()
        }
        
        return total_loss, loss_dict

class CenterLoss(nn.Module):
    """Loss for center prediction."""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
    def forward(self,
                pred_centers: torch.Tensor,
                gt_centers: torch.Tensor) -> torch.Tensor:
        """
        Compute center prediction loss.
        
        Args:
            pred_centers: (B, N, 3) predicted centers
            gt_centers: (B, N, 3) ground truth centers
            
        Returns:
            loss: Center prediction loss
        """
        if self.config['type'] == 'l2':
            loss = F.mse_loss(pred_centers, gt_centers)
        elif self.config['type'] == 'l1':
            loss = F.l1_loss(pred_centers, gt_centers)
        else:
            raise ValueError(f"Unknown center loss type: {self.config['type']}")
            
        # Apply sigma weighting if specified
        if 'sigma' in self.config:
            loss = loss / (2 * self.config['sigma']**2)
            
        return loss

class DiscriminativeLoss(nn.Module):
    """
    Discriminative loss for instance embedding learning.
    
    Reference: https://arxiv.org/abs/1708.02551
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        self.delta_v = config['delta_v']
        self.delta_d = config['delta_d']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.gamma = config['gamma']
        
    def forward(self,
                embeddings: torch.Tensor,
                instance_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute discriminative loss.
        
        Args:
            embeddings: (B, N, E) instance embeddings
            instance_labels: (B, N) instance labels
            
        Returns:
            loss: Combined discriminative loss
        """
        batch_size = embeddings.size(0)
        
        # Initialize losses
        variance_loss = 0.0
        distance_loss = 0.0
        regularization_loss = 0.0
        
        for batch_idx in range(batch_size):
            embedding = embeddings[batch_idx]  # (N, E)
            instance_label = instance_labels[batch_idx]  # (N,)
            
            # Get unique instance labels
            unique_instances = torch.unique(instance_label)
            num_instances = len(unique_instances)
            
            if num_instances <= 1:
                continue
                
            # Compute mean embeddings for each instance
            means = []
            for inst_id in unique_instances:
                if inst_id < 0:  # Skip background
                    continue
                mask = (instance_label == inst_id)
                if mask.sum() == 0:
                    continue
                mean_embedding = embedding[mask].mean(0)
                means.append(mean_embedding)
                
            if len(means) <= 1:
                continue
                
            means = torch.stack(means)  # (I, E)
            
            # Compute variance loss
            for idx, inst_id in enumerate(unique_instances):
                if inst_id < 0:
                    continue
                mask = (instance_label == inst_id)
                if mask.sum() == 0:
                    continue
                    
                instance_embeddings = embedding[mask]  # (Ni, E)
                mean = means[idx]
                
                distance = torch.norm(instance_embeddings - mean, p=2, dim=1)
                distance = torch.clamp(
                    distance - self.delta_v,
                    min=0.0
                )
                variance_loss += distance.mean()
                
            # Compute distance loss
            means = means.unsqueeze(0).expand(len(means), -1, -1)  # (I, I, E)
            means_t = means.transpose(0, 1)  # (I, I, E)
            
            dist_matrix = torch.norm(means - means_t, p=2, dim=2)  # (I, I)
            
            # Get the upper triangle of the distance matrix
            mask_2d = torch.triu(torch.ones_like(dist_matrix), diagonal=1)
            dist_matrix = dist_matrix * mask_2d
            
            # Compute distance loss
            distance = torch.clamp(
                2 * self.delta_d - dist_matrix,
                min=0.0
            )
            distance_loss += distance.sum() / (len(means) * (len(means) - 1))
            
            # Compute regularization loss
            regularization_loss += torch.norm(means, p=2, dim=1).mean()
            
        # Normalize losses by batch size
        variance_loss /= batch_size
        distance_loss /= batch_size
        regularization_loss /= batch_size
        
        # Combine losses with weights
        total_loss = (
            self.alpha * variance_loss +
            self.beta * distance_loss +
            self.gamma * regularization_loss
        )
        
        return total_loss