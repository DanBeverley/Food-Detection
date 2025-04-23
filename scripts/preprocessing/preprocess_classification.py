import json
from PIL import Image
import os
from typing import List, Dict, Tuple
from functools import lru_cache

def preprocess_classification(input_json: str, output_dir: str, size: Tuple[int, int] = (224, 224)) -> None:
    """
    Preprocess classification images for UEC Food-256 and ISIA Food-500 datasets.

    Args:
        input_json: Path to JSON metadata file.
        output_dir: Directory to save preprocessed images.
        size: Tuple of (width, height) for resizing.

    Raises:
        FileNotFoundError: If input file or images are missing.
        ValueError: If invalid data format.

    This function handles dataset-specific subfolders and caches image loading.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(input_json, 'r') as f:
            data: List[Dict[str, str]] = json.load(f)
        for item in data:
            if 'image_path' not in item or 'label' not in item:
                raise ValueError("Invalid metadata format for UEC Food-256/ISIA Food-500")
            cls = item['label']  # E.g., classes from UEC Food-256
            cls_dir = os.path.join(output_dir, cls)
            os.makedirs(cls_dir, exist_ok=True)
            @lru_cache(maxsize=100)
            def load_and_resize(path: str, sz: Tuple[int, int]) -> Image.Image:
                img = Image.open(path).convert('RGB')
                return img.resize(sz)
            img = load_and_resize(item['image_path'], size)
            fname = os.path.basename(item['image_path'])
            img.save(os.path.join(cls_dir, fname))
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"Error: {e}")
        raise

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--size', nargs=2, type=int, default=(224,224))
    args = parser.parse_args()
    preprocess_classification(args.input_json, args.output_dir, tuple(args.size))
