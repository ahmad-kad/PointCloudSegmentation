import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from typing import List, Tuple, Optional
import torch

def mean_shift_clustering(
    embeddings: np.ndarray,
    bandwidth: Optional[float] = None,
    min_bin_freq: int = 10,
    cluster_all: bool = False
) -> np.ndarray:
    """
    Perform mean shift clustering on instance embeddings.
    
    Args:
        embeddings: (N, D) array of instance embeddings
        bandwidth: Bandwidth parameter for mean shift
        min_bin_freq: Minimum frequency for bandwidth estimation
        cluster_all: Whether to cluster all points or only high-density ones
        
    Returns:
        labels: (N,) array of cluster labels
    """
    if bandwidth is None:
        # Estimate bandwidth if not provided
        bandwidth = estimate_bandwidth(
            embeddings,
            quantile=0.2,
            n_samples=1000,
            random_state=42
        )
        
    # Handle edge case where bandwidth estimation fails
    if bandwidth == 0:
        bandwidth = np.mean(np.std(embeddings, axis=0))
        
    # Perform mean shift clustering
    ms = MeanShift(
        bandwidth=bandwidth,
        bin_seeding=True,
        min_bin_freq=min_bin_freq,
        cluster_all=cluster_all,
        n_jobs=-1
    )
    
    labels = ms.fit_predict(embeddings)
    return labels

def dbscan_clustering(
    embeddings: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5
) -> np.ndarray:
    """
    Perform DBSCAN clustering on instance embeddings.
    
    Args:
        embeddings: (N, D) array of instance embeddings
        eps: Maximum distance between samples for neighborhood
        min_samples: Minimum number of samples in neighborhood
        
    Returns:
        labels: (N,) array of cluster labels
    """
    from sklearn.cluster import DBSCAN
    
    # Perform DBSCAN clustering
    db = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        n_jobs=-1
    )
    
    labels = db.fit_predict(embeddings)
    return labels

def instance_fusion(
    semantic_pred: torch.Tensor,
    instance_embeddings: torch.Tensor,
    semantic_threshold: float = 0.5,
    embedding_threshold: float = 0.7
) -> Tuple[torch.Tensor, List[int]]:
    """
    Fuse instance predictions based on semantic and embedding similarity.
    
    Args:
        semantic_pred: (N,) tensor of semantic predictions
        instance_embeddings: (N, D) tensor of instance embeddings
        semantic_threshold: Threshold for semantic similarity
        embedding_threshold: Threshold for embedding similarity
        
    Returns:
        merged_instances: (N,) tensor of merged instance labels
        valid_instances: List of valid instance IDs
    """
    device = semantic_pred.device
    N = semantic_pred.shape[0]
    
    # Initialize instance labels
    instance_labels = torch.arange(N, device=device)
    changed = True
    
    while changed:
        changed = False
        
        # Compute pairwise embedding distances
        distances = torch.cdist(
            instance_embeddings,
            instance_embeddings
        )
        
        # Compute semantic similarity
        semantic_sim = (semantic_pred.unsqueeze(0) == semantic_pred.unsqueeze(1))
        
        for i in range(N):
            for j in range(i + 1, N):
                # Check if instances should be merged
                if (semantic_sim[i, j] and 
                    distances[i, j] < embedding_threshold and 
                    instance_labels[i] != instance_labels[j]):
                    # Merge instances
                    old_label = instance_labels[j]
                    new_label = instance_labels[i]
                    instance_labels[instance_labels == old_label] = new_label
                    changed = True
                    
    # Get valid instances
    unique_instances = torch.unique(instance_labels)
    valid_instances = []
    
    for inst_id in unique_instances:
        mask = (instance_labels == inst_id)
        if torch.sum(mask) >= 10:  # Minimum instance size
            valid_instances.append(inst_id.item())
            
    return instance_labels, valid_instances