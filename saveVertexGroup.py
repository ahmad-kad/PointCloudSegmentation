import numpy as np
import open3d as o3d
import json
from pathlib import Path
import argparse
import logging
import os

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger('vertex_groups')

def find_instance_labels(ply_path):
    ply_path = Path(ply_path)
    base_name = ply_path.stem
    if base_name.endswith('_labeled'):
        base_name = base_name[:-8]  # Remove '_labeled'
    npy_path = ply_path.parent / f"{base_name}_labeled_labels.npy"
    
    if not npy_path.exists():
        raise FileNotFoundError(f"Instance labels not found at {npy_path}")
    return npy_path

def save_segmented_ply(points, instance_labels, output_path, colors=None):
    """Save a single PLY file with vertex groups for each instance"""
    output_path = output_path.with_suffix('.ply')
    unique_instances = np.unique(instance_labels)
    
    # Create vertices section
    vertices_list = []
    vertex_groups = {}
    
    # Header for PLY file
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z"
    ]
    
    if colors is not None:
        header.extend([
            "property uchar red",
            "property uchar green",
            "property uchar blue"
        ])
    
    # Add comment section for vertex groups
    vertex_groups_comments = []
    for instance_id in unique_instances:
        if instance_id == -1:
            continue
        mask = instance_labels == instance_id
        indices = np.where(mask)[0]
        vertex_groups_comments.append(f"comment vertex_group instance_{instance_id} {' '.join(map(str, indices))}")
    
    header.extend(vertex_groups_comments)
    header.append("end_header")
    
    # Write PLY file
    with open(output_path, 'w') as f:
        # Write header
        f.write('\n'.join(header) + '\n')
        
        # Write vertices
        for i in range(len(points)):
            line = f"{points[i][0]} {points[i][1]} {points[i][2]}"
            if colors is not None:
                line += f" {int(colors[i][0]*255)} {int(colors[i][1]*255)} {int(colors[i][2]*255)}"
            f.write(line + '\n')
    
    # Create summary file
    with open(output_path.with_suffix('.txt'), 'w') as f:
        f.write(f"Total points: {len(points)}\n")
        f.write(f"Number of instances: {len(unique_instances)-1}\n\n")  # -1 for background
        
        for instance_id in unique_instances:
            if instance_id == -1:
                continue
            mask = instance_labels == instance_id
            f.write(f"Instance {instance_id}:\n")
            f.write(f"  Number of points: {np.sum(mask)}\n")

def save_as_json(points, instance_labels, output_path, colors=None):
    """Save instance groups as JSON format"""
    vertex_groups = {}
    unique_instances = np.unique(instance_labels)
    
    for instance_id in unique_instances:
        if instance_id == -1:
            continue
            
        mask = instance_labels == instance_id
        point_indices = np.where(mask)[0]
        
        vertex_groups[f"instance_{instance_id}"] = {
            "indices": point_indices.tolist()
        }
    
    with open(output_path.with_suffix('.json'), 'w') as f:
        json.dump(vertex_groups, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Create vertex groups from point cloud and instance labels')
    parser.add_argument('--point-cloud', type=str, required=True,
                       help='Path to input point cloud (PLY format)')
    parser.add_argument('--output', type=str,
                       help='Output path prefix (without extension). If not specified, uses input path.')
    parser.add_argument('--format', type=str, choices=['ply', 'json'], default='ply',
                       help='Output format: ply for single file with vertex groups, json for index mapping (default: ply)')
    args = parser.parse_args()
    
    logger = setup_logging()
    
    ply_path = Path(args.point_cloud)
    output_path = Path(args.output) if args.output else ply_path.parent / f"{ply_path.stem}_segmented"
    
    try:
        npy_path = find_instance_labels(ply_path)
        logger.info(f"Found instance labels at {npy_path}")
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    logger.info(f"Loading point cloud from {ply_path}")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    
    logger.info(f"Loading instance labels from {npy_path}")
    instance_labels = np.load(npy_path)
    
    if len(points) != len(instance_labels):
        logger.error(f"Number of points ({len(points)}) does not match number of labels ({len(instance_labels)})")
        return
    
    logger.info(f"Processing {len(np.unique(instance_labels))-1} instances")  # -1 for background
    
    if args.format == 'json':
        logger.info(f"Saving results as JSON to {output_path}.json")
        save_as_json(points, instance_labels, output_path, colors)
    else:
        logger.info(f"Saving results as PLY with vertex groups to {output_path}.ply")
        save_segmented_ply(points, instance_labels, output_path, colors)
    
    logger.info("Done!")

if __name__ == "__main__":
    main()
    