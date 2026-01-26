import os
import json
import random
import argparse
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_dataset_metadata(source_rgbd_base_dir_path: str, output_metadata_dir_path: str, train_val_split_ratio: float = 0.2):
    """
    Creates instance-based train/validation split to prevent data leakage.
    All frames from a single food instance go to either train OR validation, never both.
    
    Args:
        source_rgbd_base_dir_path (str): Root directory of RGBD dataset
        output_metadata_dir_path (str): Output directory for train_metadata.json and val_metadata.json
        train_val_split_ratio (float): Validation split ratio (default: 0.2)
    """
    source_dir = Path(source_rgbd_base_dir_path)
    output_meta_dir = Path(output_metadata_dir_path)
    
    if not source_dir.is_dir():
        logging.error(f"Source directory not found: {source_dir}")
        return

    os.makedirs(output_meta_dir, exist_ok=True)

    images_by_instance = defaultdict(list)
    label_to_index = {}
    next_label_index = 0

    logging.info(f"Scanning source directory: {source_dir}")
    class_folders = [d for d in source_dir.iterdir() if d.is_dir()]
    if not class_folders:
        logging.error(f"No class subdirectories found in {source_dir}")
        return

    for class_folder in class_folders:
        class_name = class_folder.name
        instance_folders = [d for d in class_folder.iterdir() if d.is_dir()]
        
        if not instance_folders:
            continue

        if class_name not in label_to_index:
            label_to_index[class_name] = next_label_index
            next_label_index += 1

        for instance_folder in instance_folders:
            instance_name = instance_folder.name
            original_folder = instance_folder / "original"

            if not original_folder.is_dir():
                continue

            depth_folder = instance_folder / "depth"
            DEPTH_MAP_EXTENSIONS = ['.jpg', '.png'] # Added png just in case, though usually jpg in this dataset

            image_files = list(original_folder.glob('*[.jpg][.jpeg][.png][.gif]'))
            if not image_files:
                continue
            
            instance_key = f"{class_name}/{instance_name}"
            
            for img_path in image_files:
                depth_map_path = None
                if depth_folder.is_dir():
                    img_stem = img_path.stem
                    for ext in DEPTH_MAP_EXTENSIONS:
                        potential_path = depth_folder / (img_stem + ext)
                        if potential_path.is_file():
                            depth_map_path = str(potential_path.relative_to(source_dir))
                            break
                
                metadata_entry = {
                    "image_path": str(img_path.relative_to(source_dir)),
                    "depth_map_path": depth_map_path, 
                    "class_name": class_name,
                    "instance_name": instance_name
                }
                images_by_instance[instance_key].append(metadata_entry)

    if not images_by_instance:
        logging.error("No images found. Metadata files will not be created.")
        return

    # Instance-based split to prevent data leakage
    all_instance_keys = list(images_by_instance.keys())
    random.shuffle(all_instance_keys)
    
    split_point = int(len(all_instance_keys) * (1 - train_val_split_ratio))
    train_instance_keys = all_instance_keys[:split_point]
    val_instance_keys = all_instance_keys[split_point:]
    
    train_metadata = []
    for key in train_instance_keys:
        train_metadata.extend(images_by_instance[key])
    
    val_metadata = []
    for key in val_instance_keys:
        val_metadata.extend(images_by_instance[key])
    
    # Save files
    index_to_label = {v: k for k, v in label_to_index.items()}
    
    try:
        with open(output_meta_dir / "train_metadata.json", 'w') as f:
            json.dump(train_metadata, f, indent=2)
        
        with open(output_meta_dir / "val_metadata.json", 'w') as f:
            json.dump(val_metadata, f, indent=2)
        
        with open(output_meta_dir / "label_map.json", 'w') as f:
            json.dump(index_to_label, f, indent=2)
        
        logging.info(f"Instance-based split complete:")
        logging.info(f"Training instances: {len(train_instance_keys)}, Validation instances: {len(val_instance_keys)}")
        logging.info(f"Training images: {len(train_metadata)}, Validation images: {len(val_metadata)}")
        logging.info(f"Total classes: {len(label_to_index)}")

    except IOError as e:
        logging.error(f"Failed to write files: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create instance-based train/validation split to prevent data leakage.")
    parser.add_argument('--source_dir', type=str, required=True,
                        help="Root directory of dataset (structure: <ClassName>/<InstanceName>/original/<images>)")
    parser.add_argument('--output_metadata_dir', type=str, required=True,
                        help="Output directory for train_metadata.json and val_metadata.json")
    parser.add_argument('--val_split', type=float, default=0.2,
                        help="Validation split ratio (default: 0.2)")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    resolved_output_metadata_dir = Path(args.output_metadata_dir)
    if not resolved_output_metadata_dir.is_absolute():
        resolved_output_metadata_dir = project_root / args.output_metadata_dir

    create_dataset_metadata(args.source_dir, str(resolved_output_metadata_dir), args.val_split)
