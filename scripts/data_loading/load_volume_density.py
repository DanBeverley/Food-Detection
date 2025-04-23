import os
import json
from typing import List, Dict, Any

def load_volume_density_dataset(data_dir: str, output_json: str) -> List[Dict[str, Any]]:
    """
    Load volume and density measurement dataset and generate metadata list.

    Args:
        data_dir: Path to directory with sample subdirectories containing images, depth maps, and logs.
        output_json: Path to save JSON metadata file.

    Returns:
        List of dicts with keys 'sample', 'images', 'depth_maps', and 'log'.
    """
    dataset = []
    for sample in os.listdir(data_dir):
        sample_dir = os.path.join(data_dir, sample)
        if not os.path.isdir(sample_dir):
            continue
        images = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg','.png'))]
        depth_maps = [f for f in os.listdir(sample_dir) if 'depth' in f.lower()]
        logs = [f for f in os.listdir(sample_dir) if f.lower().endswith('.json')]
        dataset.append({
            'sample': sample,
            'images': [os.path.join(sample_dir, f) for f in images],
            'depth_maps': [os.path.join(sample_dir, f) for f in depth_maps],
            'log': os.path.join(sample_dir, logs[0]) if logs else None
        })
    with open(output_json, 'w') as f:
        json.dump(dataset, f, indent=2)
    return dataset

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_json', required=True)
    args = parser.parse_args()
    load_volume_density_dataset(args.data_dir, args.output_json)
