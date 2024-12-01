# Point2Net

Deep learning system for instance-level segmentation of 3D point cloud data.

## Requirements

- CUDA-capable GPU (8GB+ VRAM)
- Python 3.8+
- CUDA 11.0+
- Ubuntu 20.04+

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage


```bash
#Data Preparation
python preprocess_data.py --config configs/preprocessing.yaml --raw-dir /path/to/scannet

# Training
python train.py --config configs/training_config.yaml --data-dir data/processed

# Inference
python inference.py --checkpoint model.pth --input scene.ply
```

## Project Structure

```
point2net/
├── configs/          # Configuration files
├── data/            # Data storage
├── src/             # Source code
│   ├── models/     # Neural networks
│   ├── data/       # Data processing
│   └── utils/      # Utilities
├── outputs/         # Training outputs
└── results/         # Inference results
```

## Key Features

- Instance segmentation for 3D point clouds
- End-to-end training pipeline
- Real-time inference capability


## License

MIT License - See LICENSE file

## Citations and Acknowledgements
This project utilizes the ScanNet dataset for traini, tools, and builds upon existing research. We would like to acknowledge the following contributions:

# References

## Dataset
We use the [ScanNet](http://www.scan-net.org/) dataset for training and evaluation:
```bibtex
@inproceedings{dai2017scannet,
    title     = {ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes},
    author    = {Dai, Angela and Chang, Angel X. and Savva, Manolis and 
                 Halber, Maciej and Funkhouser, Thomas and Nie{\ss}ner, Matthias},
    booktitle = {Proc. Computer Vision and Pattern Recognition (CVPR), IEEE},
    year      = {2017}
}
```

## Tools
Experiment tracking and visualization powered by [Weights & Biases](https://wandb.com/):
```bibtex
@misc{wandb,
    title  = {Experiment Tracking with Weights and Biases},
    year   = {2020},
    note   = {Software available from wandb.com},
    url    = {https://www.wandb.com/},
    author = {Biewald, Lukas}
}
```

## Key Research
Built upon the [PointNet](https://arxiv.org/abs/1612.00593) architecture:
```bibtex
@article{qi2016pointnet,
    title   = {PointNet: Deep Learning on Point Sets for 3D Classification 
               and Segmentation},
    author  = {Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
    journal = {arXiv preprint arXiv:1612.00593},
    year    = {2016}
}

## Additional Readings
For those interested in diving deeper into 3D point cloud processing and instance segmentation, we recommend the following papers:

    PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space (Qi et al., NIPS 2017)
    https://arxiv.org/abs/1706.02413
    VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection (Zhou et al., CVPR 2018)
    https://arxiv.org/abs/1711.06396
    PointCNN: Convolution On X-Transformed Points (Li et al., NeurIPS 2018)
    https://arxiv.org/abs/1801.07791
    SGPN: Similarity Group Proposal Network for 3D Point Cloud Instance Segmentation (Wang et al., CVPR 2018)
    https://arxiv.org/abs/1711.08588
    PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud (Shi et al., CVPR 2019)
    https://arxiv.org/abs/1812.04244