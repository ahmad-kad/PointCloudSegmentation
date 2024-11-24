import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import numpy as np

class PointInstanceLoss(nn.Module):
    """
    Combination of multiple losses for instance segmentation:
    1. Feature Clustering Loss
    2. Instance Center Loss
    3. Instance Boundary Loss
    4. Lovász Loss for better handling of unbalanced instances
    """
    def __init__(
        self,
        feature_dim: int = 64,
        delta_v: float = 0.5,
        delta_d: float = 1.5,
        gamma: float = 0.001,
        weights: Dict[str, float] = None
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.gamma = gamma
        
        # Default loss weights if not provided
        self.weights = weights or {
            'clustering': 0.1,  # Reduced from 1.0
            'center': 0.1,     # Reduced from 1.0
            'boundary': 0.05,  # Reduced from 0.5
            'lovasz': 0.01      # Reduced from 1.0
        }
    def compute_lovasz_loss(
        self,
        embeddings: torch.Tensor,
        instance_labels: torch.Tensor
    ) -> torch.Tensor:
        """Modified Lovász loss with better numerical stability."""
        batch_size = embeddings.size(0)
        lovasz_loss = torch.tensor(0., device=embeddings.device)
        
        for b in range(batch_size):
            embedding = embeddings[b]
            label = instance_labels[b]
            
            unique_instances = torch.unique(label)
            if unique_instances[0] == 0:
                unique_instances = unique_instances[1:]
            
            if len(unique_instances) == 0:
                continue
            
            # Compute normalized similarities
            for instance_id in unique_instances:
                mask = (label == instance_id).float()
                if mask.sum() == 0:
                    continue
                
                # Compute normalized features similarity
                instance_embedding = embedding[mask.bool()]
                mean_embedding = instance_embedding.mean(0)
                mean_embedding = F.normalize(mean_embedding, p=2, dim=0)
                
                similarity = F.cosine_similarity(
                    embedding,
                    mean_embedding.unsqueeze(0),
                    dim=1
                )
                
                # Scale similarity to [0, 1] range
                similarity = (similarity + 1) / 2
                
                # Compute Lovász hinge loss
                sorted_sim, sorted_idx = similarity.sort(descending=True)
                sorted_mask = mask[sorted_idx]
                
                grad = self._lovasz_grad(sorted_mask)
                loss = torch.dot(F.relu(sorted_sim), grad)
                lovasz_loss += loss / len(unique_instances)
        
        return lovasz_loss / batch_size
    
    def _lovasz_grad(self, gt_sorted):
        """
        Compute Lovász gradient for sorted ground truth.
        Modified for better numerical stability.
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / (union + 1e-6)  # Add epsilon for stability
        
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        
        return jaccard
    
    def compute_clustering_loss(
        self,
        embeddings: torch.Tensor,
        instance_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Modified clustering loss with better scaling."""
        batch_size = embeddings.size(0)
        
        var_loss = torch.tensor(0., device=embeddings.device)
        dist_loss = torch.tensor(0., device=embeddings.device)
        reg_loss = torch.tensor(0., device=embeddings.device)
        
        for b in range(batch_size):
            embedding = embeddings[b]
            label = instance_labels[b]
            
            unique_instances = torch.unique(label)
            if unique_instances[0] == 0:
                unique_instances = unique_instances[1:]
            
            num_instances = len(unique_instances)
            if num_instances == 0:
                continue
            
            # Calculate cluster centers
            centers = []
            for instance_id in unique_instances:
                mask = label == instance_id
                if mask.sum() == 0:
                    continue
                center = embedding[mask].mean(0)
                centers.append(center)
            
            if not centers:
                continue
            
            centers = torch.stack(centers)
            centers = F.normalize(centers, p=2, dim=1)
            
            # Variance loss with normalized distances
            for i, instance_id in enumerate(unique_instances):
                mask = label == instance_id
                if mask.sum() == 0:
                    continue
                    
                instance_points = embedding[mask]
                instance_points = F.normalize(instance_points, p=2, dim=1)
                center = centers[i]
                
                dist = torch.norm(instance_points - center, p=2, dim=1)
                dist = torch.clamp(dist - self.delta_v, min=0)
                var_loss += dist.mean()
            
            # Distance loss between different instances
            if num_instances > 1:
                center_distances = torch.cdist(centers, centers)
                mask = torch.ones_like(center_distances, dtype=torch.bool)
                mask.fill_diagonal_(False)
                center_distances = center_distances[mask].view(-1)
                
                dist_loss += torch.clamp(
                    2 * self.delta_d - center_distances,
                    min=0
                ).mean()
            
            # Regularization
            reg_loss += torch.norm(centers, p=2).mean() * self.gamma
        
        return (
            var_loss / (batch_size + 1e-6),
            dist_loss / (batch_size + 1e-6),
            reg_loss / (batch_size + 1e-6)
        )
    
    def forward(
        self,
        points: torch.Tensor,
        embeddings: torch.Tensor,
        instance_labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=2)
        
        var_loss, dist_loss, reg_loss = self.compute_clustering_loss(embeddings, instance_labels)
        center_loss = self.compute_center_loss(points, embeddings, instance_labels)
        boundary_loss = self.compute_boundary_loss(embeddings, instance_labels)
        lovasz_loss = self.compute_lovasz_loss(embeddings, instance_labels)
        
        # Combine losses with weights and normalization
        total_loss = (
            self.weights['clustering'] * (var_loss + dist_loss + reg_loss) +
            self.weights['center'] * center_loss +
            self.weights['boundary'] * boundary_loss +
            self.weights['lovasz'] * lovasz_loss
        )
        
        return {
            'total_loss': total_loss,
            'var_loss': var_loss,
            'dist_loss': dist_loss,
            'reg_loss': reg_loss,
            'center_loss': center_loss,
            'boundary_loss': boundary_loss,
            'lovasz_loss': lovasz_loss
        }
    def compute_center_loss(
        self,
        points: torch.Tensor,
        embeddings: torch.Tensor,
        instance_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss based on distance to instance centers in 3D space.
        This helps in learning better spatial relationships.
        """
        batch_size = points.size(0)
        center_loss = torch.tensor(0., device=points.device)
        
        for b in range(batch_size):
            point_coords = points[b]  # (N, 3)
            embedding = embeddings[b]  # (N, D)
            label = instance_labels[b]  # (N,)
            
            unique_instances = torch.unique(label)
            if unique_instances[0] == 0:
                unique_instances = unique_instances[1:]
            
            for instance_id in unique_instances:
                mask = label == instance_id
                if mask.sum() == 0:
                    continue
                
                # Calculate geometric center
                instance_points = point_coords[mask]
                geometric_center = instance_points.mean(0)
                
                # Calculate feature center
                instance_features = embedding[mask]
                feature_center = instance_features.mean(0)
                
                # Distance between points and their geometric center
                spatial_distances = torch.norm(instance_points - geometric_center, p=2, dim=1)
                
                # Distance between features and their center
                feature_distances = torch.norm(instance_features - feature_center, p=2, dim=1)
                
                # Combine distances with normalization
                center_loss += (spatial_distances * feature_distances).mean()
        
        return center_loss / batch_size
    
    def compute_boundary_loss(
        self,
        embeddings: torch.Tensor,
        instance_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute boundary-aware loss to better handle instance boundaries.
        Uses feature gradients to detect and enhance boundary regions.
        """
        batch_size = embeddings.size(0)
        boundary_loss = torch.tensor(0., device=embeddings.device)
        
        for b in range(batch_size):
            embedding = embeddings[b]  # (N, D)
            label = instance_labels[b]  # (N,)
            
            # Compute pairwise distances in feature space
            dist_matrix = torch.cdist(embedding, embedding)  # (N, N)
            
            # Create instance boundary mask
            instance_matrix = label.unsqueeze(0) == label.unsqueeze(1)  # (N, N)
            boundary_mask = ~instance_matrix
            
            # Compute boundary loss
            boundary_dists = dist_matrix * boundary_mask.float()
            boundary_loss += torch.clamp(self.delta_d - boundary_dists, min=0).mean()
        
        return boundary_loss / batch_size
    
    def compute_lovasz_loss(
        self,
        embeddings: torch.Tensor,
        instance_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Lovász loss for better handling of unbalanced instances.
        Lovász extension of IoU loss is particularly effective for segmentation.
        """
        def lovasz_grad(gt_sorted):
            """Compute Lovász gradient for sorted ground truth."""
            p = len(gt_sorted)
            gts = gt_sorted.sum()
            intersection = gts - gt_sorted.float().cumsum(0)
            union = gts + (1 - gt_sorted).float().cumsum(0)
            jaccard = 1. - intersection / union
            if p > 1:
                jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
            return jaccard
        
        batch_size = embeddings.size(0)
        lovasz_loss = torch.tensor(0., device=embeddings.device)
        
        for b in range(batch_size):
            embedding = embeddings[b]
            label = instance_labels[b]
            
            unique_instances = torch.unique(label)
            if unique_instances[0] == 0:
                unique_instances = unique_instances[1:]
            
            for instance_id in unique_instances:
                mask = (label == instance_id).float()
                if mask.sum() == 0:
                    continue
                
                # Compute features similarity
                instance_embedding = embedding[mask.bool()]
                mean_embedding = instance_embedding.mean(0)
                similarity = F.cosine_similarity(embedding, mean_embedding.unsqueeze(0))
                
                # Sort and calculate Lovász loss
                sorted_sim, sorted_idx = similarity.sort(descending=True)
                sorted_mask = mask[sorted_idx]
                grad = lovasz_grad(sorted_mask)
                lovasz_loss += (F.relu(sorted_sim) * grad).sum()
        
        return lovasz_loss / batch_size


class InstanceSegLoss(nn.Module):
    def __init__(
        self,
        feature_dim: int = 64,
        delta_v: float = 0.1,
        delta_d: float = 0.5
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.delta_v = delta_v
        self.delta_d = delta_d
        
        self.instance_loss = DiscriminativeLoss(
            delta_v=delta_v,
            delta_d=delta_d
        )
        self.smoothness_loss = SmoothnessLoss()
    
    def forward(
        self,
        points: torch.Tensor,
        embeddings: torch.Tensor,
        instance_labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=2)
        
        # Instance discrimination loss
        instance_loss = self.instance_loss(embeddings, instance_labels)
        
        # Spatial smoothness loss
        smoothness_loss = self.smoothness_loss(points, embeddings)
        
        total_loss = instance_loss['total_loss'] + 0.1 * smoothness_loss
        
        return {
            'total_loss': total_loss,
            'instance_loss': instance_loss['total_loss'],
            'smoothness_loss': smoothness_loss,
            'pull_loss': instance_loss['pull_loss'],
            'push_loss': instance_loss['push_loss']
        }

class DiscriminativeLoss(nn.Module):
    def __init__(self, delta_v=0.1, delta_d=0.5):
        super().__init__()
        self.delta_v = delta_v
        self.delta_d = delta_d
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = embeddings.size(0)
        
        pull_loss = torch.tensor(0., device=embeddings.device)
        push_loss = torch.tensor(0., device=embeddings.device)
        
        for b in range(batch_size):
            embedding = embeddings[b]  # (N, D)
            label = labels[b]  # (N,)
            
            # Get unique instance labels (excluding background)
            unique_labels = torch.unique(label)
            unique_labels = unique_labels[unique_labels != 0]
            
            if len(unique_labels) == 0:
                continue
            
            # Calculate mean embeddings for each instance
            means = []
            for instance_id in unique_labels:
                mask = label == instance_id
                if mask.sum() == 0:
                    continue
                mean = embedding[mask].mean(0)
                means.append(mean)
            
            if not means:
                continue
            
            means = torch.stack(means)  # (I, D)
            num_instances = means.size(0)
            
            # Pull loss: variance within each instance
            instance_pull_loss = torch.tensor(0., device=embeddings.device)
            for i, instance_id in enumerate(unique_labels):
                mask = label == instance_id
                if mask.sum() == 0:
                    continue
                
                instance_embeddings = embedding[mask]  # (Ni, D)
                mean = means[i]  # (D,)
                
                # Calculate distance to mean
                dist = torch.norm(instance_embeddings - mean.unsqueeze(0), p=2, dim=1)
                dist = torch.clamp(dist - self.delta_v, min=0)
                instance_pull_loss += dist.mean()
            
            pull_loss += instance_pull_loss / (num_instances + 1e-6)
            
            # Push loss: distance between different instances
            if num_instances > 1:
                means = F.normalize(means, p=2, dim=1)  # Normalize instance centers
                distances = torch.cdist(means, means)  # (I, I)
                
                # Create mask to consider only unique pairs
                mask = torch.triu(torch.ones_like(distances), diagonal=1)
                
                # Calculate push loss
                dist = 2 * self.delta_d - distances
                dist = torch.clamp(dist, min=0) * mask
                push_loss += dist.sum() / (mask.sum() + 1e-6)
        
        pull_loss = pull_loss / batch_size
        push_loss = push_loss / batch_size
        
        total_loss = pull_loss + push_loss
        
        return {
            'total_loss': total_loss,
            'pull_loss': pull_loss,
            'push_loss': push_loss
        }

class SmoothnessLoss(nn.Module):
    def __init__(self, k: int = 20):
        super().__init__()
        self.k = k
    
    def forward(self, points: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial smoothness loss using k-nearest neighbors.
        
        Args:
            points: (B, N, 3) point coordinates
            embeddings: (B, N, D) point embeddings
        """
        batch_size = points.size(0)
        smoothness_loss = torch.tensor(0., device=points.device)
        
        for b in range(batch_size):
            point = points[b]  # (N, 3)
            embedding = embeddings[b]  # (N, D)
            
            # Compute pairwise distances between points
            dist = torch.cdist(point, point)  # (N, N)
            
            # Get k-nearest neighbors
            _, idx = dist.topk(k=self.k + 1, dim=1, largest=False)
            idx = idx[:, 1:]  # Exclude self
            
            # Compute embedding differences with neighbors
            neighbor_embeddings = embedding[idx]  # (N, k, D)
            embedding_expanded = embedding.unsqueeze(1).expand(-1, self.k, -1)  # (N, k, D)
            
            # Compute smoothness loss
            diff = torch.norm(neighbor_embeddings - embedding_expanded, p=2, dim=2)
            smoothness_loss += diff.mean()
        
        return smoothness_loss / batch_size


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in instance segmentation.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss for instance segmentation.
        
        Args:
            predictions: (B, N, D) embeddings
            targets: (B, N) instance labels
        """
        batch_size, num_points = targets.size()
        device = predictions.device

        # Convert instance labels to one-hot encoding
        unique_instances = torch.unique(targets)
        num_instances = len(unique_instances)
        
        # Create one-hot encoded target
        one_hot_target = torch.zeros(
            (batch_size, num_points, num_instances),
            device=device
        )
        
        for i, instance_id in enumerate(unique_instances):
            mask = (targets == instance_id)
            one_hot_target[..., i][mask] = 1
        
        # Compute similarity scores
        predictions = F.normalize(predictions, p=2, dim=2)
        similarity = torch.matmul(predictions, predictions.transpose(1, 2))
        
        # Apply focal loss
        ce_loss = F.binary_cross_entropy_with_logits(
            similarity,
            one_hot_target,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        return focal_loss.mean()
    


