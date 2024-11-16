from src.data import ScanNetDataset

# Initialize dataset
dataset = ScanNetDataset(
    root_dir='data/processed',
    split='train',
    config_path='configs/data_config.yaml'
)

# Check first sample
sample = dataset[0]
print("Data sample contents:", sample.keys())
print("Points shape:", sample['points'].shape)
print("Number of instances:", len(torch.unique(sample['instance_labels'])))