from typing import List, Dict, Tuple
from PIL import Image
import os
import json
from functools import lru_cache

def preprocess_segmentation(input_json: str, output_images_dir: str, output_masks_dir: str, size: Tuple[int, int] = (512, 512)) -> None:
    """
    Preprocess segmentation images and masks for datasets like FoodPix Complete, COCO, and OpenImages.

    Args:
        input_json: Path to JSON metadata file.
        output_images_dir: Directory to save preprocessed images.
        output_masks_dir: Directory to save preprocessed masks.
        size: Tuple of (width, height) for resizing.

    Raises:
        FileNotFoundError: If input file or images/masks are missing.
        ValueError: If metadata format is invalid.
        IOError: If file operations fail.

    This function uses caching for image loading, handles dataset-specific structures (e.g., subfolders for classes),
    and includes error handling for robustness.
    """
    try:
        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_masks_dir, exist_ok=True)
        with open(input_json, 'r') as f:
            data: List[Dict[str, str]] = json.load(f)
        
        for item in data:
            if 'image_path' not in item or 'mask_path' not in item or 'label' not in item:
                raise ValueError("Invalid metadata format for FoodPix/COCO/OpenImages")
            
            cls = item['label']  # E.g., classes from FoodPix or COCO
            img_dir = os.path.join(output_images_dir, cls)
            mask_dir = os.path.join(output_masks_dir, cls)
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)
            
            @lru_cache(maxsize=100)
            def load_and_resize_image(path: str, sz: Tuple[int, int]) -> Image.Image:
                img = Image.open(path).convert('RGB' if 'image' in path else 'L')
                return img.resize(sz)
            
            img = load_and_resize_image(item['image_path'], size)
            mask = load_and_resize_image(item['mask_path'], size)  # Masks are grayscale
            
            fname = os.path.basename(item['image_path'])
            img.save(os.path.join(img_dir, fname))
            mask.save(os.path.join(mask_dir, fname))
    
    except (FileNotFoundError, json.JSONDecodeError, ValueError, IOError) as e:
        print(f"Error: {e}")
        raise

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess segmentation datasets.")
    parser.add_argument('--input_json', required=True, help="Path to input JSON metadata.")
    parser.add_argument('--output_images_dir', required=True, help="Directory for output images.")
    parser.add_argument('--output_masks_dir', required=True, help="Directory for output masks.")
    parser.add_argument('--size', nargs=2, type=int, default=[512, 512], help="Resize dimensions (width height).")
    args = parser.parse_args()
    preprocess_segmentation(args.input_json, args.output_images_dir, args.output_masks_dir, tuple(args.size))