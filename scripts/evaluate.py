import torch
import argparse
import yaml
import wandb
from pathlib import Path
from tqdm import tqdm

from src.models import PointNet2InstanceSegmentation, SparseCNNInstanceSegmentation
from src.data import ScanNetDataModule
from src.evaluation import evaluate_predictions
from src.evaluation.visualizer import PointCloudVisualizer
from src.utils.logger import Logger

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate instance segmentation model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model', type=str, choices=['pointnet2', 'sparsecnn'], required=True)
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Set up logging
    logger = Logger(
        name='evaluation',
        save_dir=args.output_dir,
        wandb_config=config['logging']['wandb'] if config['logging']['wandb']['enabled'] else None
    )
    
    # Initialize data module
    data_module = ScanNetDataModule(
        data_dir=config['dataset']['root_dir'],
        config_path=args.config,
        batch_size=config['training']['batch_size']['test'],
        num_workers=config['hardware']['num_workers']
    )
    data_module.setup(stage='test')
    
    # Initialize model
    device = torch.device(config['hardware']['device'])
    if args.model == 'pointnet2':
        model = PointNet2InstanceSegmentation(config['model'])
    else:
        model = SparseCNNInstanceSegmentation(config['model'])
        
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Initialize visualizer if needed
    visualizer = PointCloudVisualizer(config) if args.visualize else None
    
    # Evaluation loop
    all_metrics = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_module.test_dataloader())):
            # Move data to device
            batch = {k: v.to(device) if v is not None else None 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch['points'], batch.get('features'))
            
            # Compute metrics
            metrics = evaluate_predictions(
                outputs,
                batch,
                config['evaluation']
            )
            all_metrics.append(metrics)
            
            # Generate visualizations
            if args.visualize and batch_idx < config['logging']['visualization']['num_samples']:
                visualizer.visualize_predictions(
                    points=batch['points'][0],
                    instance_labels=outputs['instance_predictions'][0],
                    semantic_labels=outputs['semantic_logits'].argmax(dim=2)[0],
                    filename=f'sample_{batch_idx}'
                )
                
                # Visualize embeddings
                visualizer.plot_embeddings(
                    embeddings=outputs['instance_embeddings'][0],
                    instance_labels=batch['instance_labels'][0],
                    filename=f'embeddings_{batch_idx}'
                )
    
    # Compute average metrics
    mean_metrics = {
        k: np.mean([m[k] for m in all_metrics])
        for k in all_metrics[0].keys()
    }
    
    # Log results
    logger.log_metrics(mean_metrics, step=0)
    logger.info(f"Evaluation Results:")
    for k, v in mean_metrics.items():
        logger.info(f"{k}: {v:.4f}")
        
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    with open(output_dir / 'evaluation_results.yaml', 'w') as f:
        yaml.dump(mean_metrics, f)
        
    logger.finish()

if __name__ == '__main__':
    main()