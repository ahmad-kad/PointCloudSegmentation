import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet2Encoder(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        
        self.mlp1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x: (B, N, C) -> (B, C, N)
        x = x.transpose(1, 2)
        
        # Encode features
        f1 = self.mlp1(x)
        f2 = self.mlp2(f1)
        f3 = self.mlp3(f2)
        
        # Global feature
        global_f = torch.max(f3, dim=2)[0]  # (B, 256)
        
        return f1, f2, f3, global_f

class Point2Net(nn.Module):
    def __init__(self, feature_dim=6, embedding_dim=64):
        super().__init__()
        
        self.encoder = PointNet2Encoder(feature_dim)
        
        # Instance embedding network
        self.embedding_net = nn.Sequential(
            nn.Conv1d(256 + 256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, embedding_dim, 1)
        )
        
    def forward(self, x):
        """
        Args:
            x: Point cloud with features (B, N, C)
        Returns:
            embeddings: Point embeddings (B, N, embedding_dim)
        """
        # Encode features
        f1, f2, f3, global_f = self.encoder(x)
        
        # Expand global feature and concatenate
        global_f = global_f.unsqueeze(2).expand(-1, -1, f3.size(2))
        combined = torch.cat([f3, global_f], dim=1)
        
        # Generate embeddings
        embeddings = self.embedding_net(combined)
        
        # (B, embedding_dim, N) -> (B, N, embedding_dim)
        embeddings = embeddings.transpose(1, 2)
        
        return embeddings

class DiscriminativeLoss(nn.Module):
    def __init__(self, delta_v=0.5, delta_d=1.5, alpha=1.0, beta=1.0, gamma=0.001):
        super().__init__()
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: Point embeddings (B, N, E)
            labels: Instance labels (B, N)
        """
        batch_size = embeddings.size(0)
        
        loss = 0
        for b in range(batch_size):
            embedding = embeddings[b]  # (N, E)
            label = labels[b]  # (N,)
            
            # Get unique instances
            unique_labels = torch.unique(label)
            if unique_labels[0] == 0:  # Remove background
                unique_labels = unique_labels[1:]
            
            num_instances = len(unique_labels)
            if num_instances == 0:
                continue
            
            # Calculate cluster centers
            centers = []
            for instance_id in unique_labels:
                mask = label == instance_id
                center = embedding[mask].mean(0)
                centers.append(center)
            
            centers = torch.stack(centers)  # (I, E)
            
            # Variance term
            variance_term = 0
            for i, instance_id in enumerate(unique_labels):
                mask = label == instance_id
                instance_embeddings = embedding[mask]
                center = centers[i]
                
                distance = torch.norm(instance_embeddings - center, p=2, dim=1)
                distance = torch.clamp(distance - self.delta_v, min=0)
                variance_term += distance.mean()
            
            variance_term /= num_instances
            
            # Distance term
            distance_term = 0
            for i in range(num_instances):
                for j in range(i + 1, num_instances):
                    center_distance = torch.norm(centers[i] - centers[j], p=2)
                    distance = torch.clamp(2 * self.delta_d - center_distance, min=0)
                    distance_term += distance
            
            if num_instances > 1:
                distance_term /= (num_instances * (num_instances - 1)) / 2
            
            # Regularization term
            reg_term = torch.norm(centers, p=2, dim=1).mean()
            
            # Combined loss
            instance_loss = self.alpha * variance_term + self.beta * distance_term + self.gamma * reg_term
            loss += instance_loss
        
        return loss / batch_size