import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import average_precision_score
from .clustering import mean_shift_clustering

def evaluate_predictions(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    config: dict
) -> Dict[str, float]:
    """
    Evaluate instance segmentation predictions.
    
    Args:
        predictions: Dict containing model predictions
        targets: Dict containing ground truth
        config: Evaluation configuration
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Get instance predictions through clustering
    instance_preds = get_instances(
        predictions,
        config['validation']['score_threshold'],
        config['validation']['nms_threshold']
    )
    
    # Calculate metrics
    metrics = {}
    
    # Mean Average Precision
    map_score = calculate_map(
        instance_preds,
        targets['instance_labels'],
        targets['semantic_labels'],
        iou_threshold=0.5
    )
    metrics['mAP'] = map_score
    
    # Instance-level IoU
    iou_score = calculate_instance_iou(
        instance_preds,
        targets['instance_labels']
    )
    metrics['IoU'] = iou_score
    
    # Semantic segmentation accuracy
    sem_acc = calculate_semantic_accuracy(
        predictions['semantic_logits'],
        targets['semantic_labels']
    )
    metrics['semantic_accuracy'] = sem_acc
    
    return metrics

def get_instances(
    predictions: Dict[str, torch.Tensor],
    score_threshold: float,
    nms_threshold: float
) -> torch.Tensor:
    """
    Get instance predictions through clustering.
    
    Args:
        predictions: Model predictions
        score_threshold: Confidence threshold
        nms_threshold: NMS IoU threshold
        
    Returns:
        instance_predictions: Instance ID for each point
    """
    batch_size = predictions['semantic_logits'].shape[0]
    device = predictions['semantic_logits'].device
    
    instance_predictions = []
    
    for b in range(batch_size):
        # Get semantic predictions
        sem_logits = predictions['semantic_logits'][b]
        sem_probs = torch.softmax(sem_logits, dim=1)
        sem_labels = torch.argmax(sem_probs, dim=1)
        
        # Get embeddings
        embeddings = predictions['instance_embeddings'][b]
        
        # Get center predictions
        centers = predictions['center_predictions'][b]
        
        # Initialize instance IDs
        instance_ids = torch.zeros_like(sem_labels)
        
        # Process each semantic class separately
        for sem_id in torch.unique(sem_labels):
            if sem_id == 0:  # Skip background
                continue
                
            # Get points belonging to current class
            mask = (sem_labels == sem_id)
            if not mask.any():
                continue
                
            # Get class-specific embeddings
            class_embeddings = embeddings[mask]
            
            # Perform mean shift clustering
            cluster_labels = mean_shift_clustering(
                class_embeddings.cpu().numpy(),
                bandwidth=0.6  # Adjust based on your needs
            )
            
            # Assign instance IDs
            instance_ids[mask] = cluster_labels + 1 + \
                               (sem_id * 1000)  # Offset to keep instances separate
                               
        instance_predictions.append(instance_ids)
        
    return torch.stack(instance_predictions)

def calculate_map(
    pred_instances: torch.Tensor,
    gt_instances: torch.Tensor,
    gt_semantics: torch.Tensor,
    iou_threshold: float = 0.5
) -> float:
    """Calculate mean average precision."""
    batch_size = pred_instances.shape[0]
    total_map = 0.0
    
    for b in range(batch_size):
        pred_inst = pred_instances[b]
        gt_inst = gt_instances[b]
        gt_sem = gt_semantics[b]
        
        # Get unique instances
        pred_ids = torch.unique(pred_inst)
        gt_ids = torch.unique(gt_inst)
        
        if len(pred_ids) == 1 or len(gt_ids) == 1:
            continue
            
        # Calculate IoU matrix
        ious = []
        for p_id in pred_ids:
            if p_id == 0:  # Skip background
                continue
            p_mask = (pred_inst == p_id)
            
            instance_ious = []
            for g_id in gt_ids:
                if g_id == 0:
                    continue
                g_mask = (gt_inst == g_id)
                
                intersection = torch.sum(p_mask & g_mask).float()
                union = torch.sum(p_mask | g_mask).float()
                iou = intersection / (union + 1e-6)
                
                instance_ious.append(iou.item())
            
            ious.append(max(instance_ious))
            
        # Calculate AP
        y_true = [1 if iou >= iou_threshold else 0 for iou in ious]
        y_score = [iou for iou in ious]
        
        if len(y_true) > 0:
            ap = average_precision_score(y_true, y_score)
            total_map += ap
            
    return total_map / batch_size

def calculate_instance_iou(
    pred_instances: torch.Tensor,
    gt_instances: torch.Tensor
) -> float:
    """Calculate mean instance-level IoU."""
    batch_size = pred_instances.shape[0]
    total_iou = 0.0
    
    for b in range(batch_size):
        pred_inst = pred_instances[b]
        gt_inst = gt_instances[b]
        
        # Get unique instances
        pred_ids = torch.unique(pred_inst)
        gt_ids = torch.unique(gt_inst)
        
        if len(pred_ids) == 1 or len(gt_ids) == 1:
            continue
            
        # Calculate IoU for each predicted instance
        instance_ious = []
        for p_id in pred_ids:
            if p_id == 0:
                continue
            p_mask = (pred_inst == p_id)
            
            # Find best matching ground truth instance
            best_iou = 0.0
            for g_id in gt_ids:
                if g_id == 0:
                    continue
                g_mask = (gt_inst == g_id)
                
                intersection = torch.sum(p_mask & g_mask).float()
                union = torch.sum(p_mask | g_mask).float()
                iou = intersection / (union + 1e-6)
                
                best_iou = max(best_iou, iou.item())
                
            instance_ious.append(best_iou)
            
        if len(instance_ious) > 0:
            total_iou += np.mean(instance_ious)
            
    return total_iou / batch_size

def calculate_semantic_accuracy(
    semantic_logits: torch.Tensor,
    semantic_labels: torch.Tensor
) -> float:
    """Calculate semantic segmentation accuracy."""
    predictions = torch.argmax(semantic_logits, dim=1)
    correct = (predictions == semantic_labels).float()
    accuracy = torch.mean(correct).item()
    
    return accuracy