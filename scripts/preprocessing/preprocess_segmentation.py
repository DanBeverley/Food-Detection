import json
import os
from PIL import Image

def preprocess_segmentation(input_json, output_images_dir, output_masks_dir, size=(512,512)):
    """
    Resize segmentation images and masks; save aligned to separate dirs.
    """
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)
    with open(input_json) as f:
        data = json.load(f)
    for item in data:
        img = Image.open(item['image_path']).convert('RGB').resize(size)
        mask = Image.open(item['mask_path']).convert('L').resize(size)
        fname = os.path.basename(item['image_path'])
        cls = item['label']
        img_dir = os.path.join(output_images_dir, cls)
        mask_dir = os.path.join(output_masks_dir, cls)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        img.save(os.path.join(img_dir, fname))
        mask.save(os.path.join(mask_dir, fname))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', required=True)
    parser.add_argument('--output_images_dir', required=True)
    parser.add_argument('--output_masks_dir', required=True)
    parser.add_argument('--size', nargs=2, type=int, default=(512,512))
    args = parser.parse_args()
    preprocess_segmentation(args.input_json, args.output_images_dir, args.output_masks_dir, tuple(args.size))
