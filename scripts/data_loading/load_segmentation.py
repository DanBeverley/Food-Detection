import os
import json
from PIL import Image
from typing import List, Dict

def load_segmentation_dataset(data_dir: str, output_json: str) -> List[Dict[str, str]]:
    """
    Load segmentation images and masks, generate metadata list.

    Args:
        data_dir: Path to root directory with class subdirectories, each containing 'images/' and 'masks/'.
        output_json: Path to save JSON metadata file.

    Returns:
        List of dicts with keys 'image_path', 'mask_path', and 'label'.
    """
    classes = os.listdir(data_dir)
    dataset = []
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        images_dir = os.path.join(cls_dir, 'images')
        masks_dir = os.path.join(cls_dir, 'masks')
        for fname in os.listdir(images_dir):
            if fname.lower().endswith(('.jpg', '.png')):
                dataset.append({
                    'image_path': os.path.join(images_dir, fname),
                    'mask_path': os.path.join(masks_dir, fname),
                    'label': cls
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
    load_segmentation_dataset(args.data_dir, args.output_json)
