import os
import shutil
import argparse
import logging
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_polygons_from_mask(mask_path):
    """
    Reads a binary mask and returns normalized polygons for YOLO format.
    Format: [x1, y1, x2, y2, ...] normalized to [0, 1]
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    
    height, width = mask.shape
    
    # Find contours
    # RETR_EXTERNAL: retrieves only the extreme outer contours
    # CHAIN_APPROX_SIMPLE: compresses horizontal, vertical, and diagonal segments
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < 50:  # Filter small noise
            continue
            
        # Flatten contour coordinates
        contour = contour.flatten().astype(float)
        
        # Normalize coordinates
        # Contour format: x1, y1, x2, y2, ...
        # x is at even indices (0, 2, ...), y at odd indices (1, 3, ...)
        contour[0::2] /= width
        contour[1::2] /= height
        
        # Clip to [0, 1] just in case
        contour = np.clip(contour, 0.0, 1.0)
        
        polygons.append(contour.tolist())
        
    return polygons

def create_yolo_dataset(source_dir_path, output_dir_path, val_split=0.2):
    source_dir = Path(source_dir_path)
    output_dir = Path(output_dir_path)
    
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    
    # Create YOLO directory structure
    for split in ['train', 'val']:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)
        
    # Scan for classes
    class_folders = [d for d in source_dir.iterdir() if d.is_dir()]
    if not class_folders:
        logging.error(f"No class folders found in {source_dir}")
        return

    # Create Label Map
    class_names = sorted([d.name for d in class_folders])
    label_map = {name: idx for idx, name in enumerate(class_names)}
    
    # Save dataset.yaml for YOLO training
    yaml_content = f"""path: {output_dir.resolve().as_posix()} # dataset root dir
train: images/train
val: images/val

# Classes
names:
"""
    for idx, name in enumerate(class_names):
        yaml_content += f"  {idx}: {name}\n"
        
    with open(output_dir / "dataset.yaml", 'w') as f:
        f.write(yaml_content)
    logging.info(f"Created dataset.yaml at {output_dir / 'dataset.yaml'}")

    all_entries = []
    
    logging.info("Scanning dataset and converting masks...")
    
    for class_folder in class_folders:
        class_name = class_folder.name
        class_id = label_map[class_name]
        
        instance_folders = [d for d in class_folder.iterdir() if d.is_dir()]
        
        for instance_folder in instance_folders:
            original_folder = instance_folder / "original"
            mask_folder = instance_folder / "masks"
            
            if not original_folder.is_dir() or not mask_folder.is_dir():
                continue
                
            image_files = list(original_folder.glob('*[.jpg][.jpeg][.png]'))
            
            for img_path in image_files:
                # Find corresponding mask
                mask_path = mask_folder / (img_path.stem + ".png") # Assuming .png for masks
                if not mask_path.exists():
                     # Try with suffix if needed, but assuming standard name match from previous checks
                     continue
                
                all_entries.append({
                    'image_path': img_path,
                    'mask_path': mask_path,
                    'class_id': class_id,
                    'base_name': f"{class_name}_{instance_folder.name}_{img_path.name}"
                })

    # Split dataset
    train_entries, val_entries = train_test_split(all_entries, test_size=val_split, random_state=42)
    
    def process_entries(entries, split_name):
        for entry in tqdm(entries, desc=f"Processing {split_name}"):
            # 1. Get Polygons
            polygons = get_polygons_from_mask(entry['mask_path'])
            if not polygons:
                continue # Skip empty masks
                
            # 2. Write Label File
            # YOLO format: <class_id> <x1> <y1> ...
            label_filename = Path(entry['base_name']).with_suffix('.txt')
            label_path = labels_dir / split_name / label_filename
            
            with open(label_path, 'w') as f:
                for poly in polygons:
                    poly_str = " ".join([f"{coord:.6f}" for coord in poly])
                    f.write(f"{entry['class_id']} {poly_str}\n")
            
            # 3. Copy Image
            # Use unique name to avoid collisions
            dest_img_path = images_dir / split_name / entry['base_name']
            shutil.copy2(entry['image_path'], dest_img_path)

    process_entries(train_entries, 'train')
    process_entries(val_entries, 'val')
    
    logging.info(f"YOLOv8 dataset preparation complete: {len(train_entries)} train, {len(val_entries)} val images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MetaFood3D to YOLOv8 segmentation format")
    parser.add_argument("--source_dir", required=True, help="Path to raw RGBD_videos directory")
    parser.add_argument("--output_dir", required=True, help="Path to output YOLO dataset")
    parser.add_argument("--val_split", type=float, default=0.2)
    
    args = parser.parse_args()
    create_yolo_dataset(args.source_dir, args.output_dir, args.val_split)
