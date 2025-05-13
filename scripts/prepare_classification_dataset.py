import os
import json
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_dataset_metadata(source_rgbd_base_dir_path: str, output_metadata_dir_path: str, train_val_split_ratio_for_log: float = 0.2):
    """
    Scans a source directory of images organized into class/instance/original subfolders,
    and generates a metadata.json file with absolute paths to the original images.
    No images are copied; this script only creates the metadata file.

    The expected source structure is: source_rgbd_base_dir_path/<ClassName>/<InstanceName>/original/<image_files>

    Args:
        source_rgbd_base_dir_path (str): The root directory of the original RGBD dataset 
                                         (e.g., 'E:\\_MetaFood3D_new_RGBD_videos\\RGBD_videos').
        output_metadata_dir_path (str): The directory where 'metadata.json' will be created
                                       (e.g., 'data/MetaFood3D_RGBD_classification').
        train_val_split_ratio_for_log (float): The proportion of the dataset to include in the validation split for logging purposes.
                                                Actual data splitting for TF Datasets is handled by data.py based on config.
    """
    source_dir = Path(source_rgbd_base_dir_path)
    output_meta_dir = Path(output_metadata_dir_path)
    metadata_file_path = output_meta_dir / "metadata.json"

    if not source_dir.is_dir():
        logging.error(f"Source directory not found: {source_dir}")
        return

    os.makedirs(output_meta_dir, exist_ok=True) # Ensure directory for metadata.json exists

    all_metadata_entries = []
    class_counts = {}
    label_to_index = {} # To store class name to integer mapping
    next_label_index = 0 # Counter for assigning new labels
    total_images_referenced = 0

    logging.info(f"Scanning source directory: {source_dir} to generate metadata (no images will be copied).")
    class_folders = [d for d in source_dir.iterdir() if d.is_dir()]
    if not class_folders:
        logging.error(f"No class subdirectories found in {source_dir}")
        return

    logging.info(f"Found {len(class_folders)} potential class folders: {[cf.name for cf in class_folders]}")

    for class_folder in class_folders:
        class_name = class_folder.name
        class_image_count = 0

        instance_folders = [d for d in class_folder.iterdir() if d.is_dir()]
        if not instance_folders:
            logging.warning(f"No instance subdirectories found for class '{class_name}' in {class_folder}")
            continue

        logging.info(f"Processing class '{class_name}': Found {len(instance_folders)} instance folders.")

        for instance_folder in instance_folders:
            instance_name = instance_folder.name # Retain for potential future use, but not for image naming now
            original_folder = instance_folder / "original"

            if not original_folder.is_dir():
                logging.warning(f"  No 'original' folder found in instance '{instance_name}' for class '{class_name}'. Skipping.")
                continue

            # Ensure class_name is in label_to_index map
            if class_name not in label_to_index: # Check and assign new label
                label_to_index[class_name] = next_label_index
                next_label_index += 1

            image_files = list(original_folder.glob('*[.jpg][.jpeg][.png][.gif]'))
            if not image_files:
                continue
            
            for img_path in image_files:
                try:
                    # Reference the original image directly
                    metadata_entry = {
                        "image_path": str(img_path.resolve()), # Absolute path to original image
                        "class_name": class_name,
                        "instance_name": instance_folder.name
                    }
                    all_metadata_entries.append(metadata_entry)
                    class_image_count += 1
                    total_images_referenced += 1
                except Exception as e:
                    logging.error(f"Error processing image path {img_path}: {e}")
        
        if class_image_count > 0:
            class_counts[class_name] = class_image_count
            logging.info(f"Completed referencing images for class '{class_name}', total images: {class_image_count}")
        else:
            logging.warning(f"No images referenced for class '{class_name}' after checking all its instances.")

    if not all_metadata_entries:
        logging.error("No images were referenced. Metadata file will not be created.")
        return

    # Create index_to_label for saving the label_map.json in the desired format
    index_to_label = {v: k for k, v in label_to_index.items()}
    label_map_file_path = output_meta_dir / "label_map.json" # Path for label_map.json

    try:
        with open(metadata_file_path, 'w') as f:
            json.dump(all_metadata_entries, f, indent=2)
        logging.info(f"Successfully created metadata file at: {metadata_file_path}")

        with open(label_map_file_path, 'w') as f: # Save label_map.json
            json.dump(index_to_label, f, indent=2)
        logging.info(f"Successfully created label map file at: {label_map_file_path}")
        logging.info(f"Total unique classes found: {len(label_to_index)}")

        logging.info(f"Total images referenced overall: {total_images_referenced}")
        logging.info("Image counts per class in the metadata file:")
        for class_name, count in class_counts.items():
            approx_train = int(count * (1 - train_val_split_ratio_for_log))
            approx_val = int(count * train_val_split_ratio_for_log)
            logging.info(f"  {class_name}: {count} (For reference: approx. train {approx_train}, approx. val {approx_val} based on {train_val_split_ratio_for_log*100}% val split)")

    except IOError as e:
        logging.error(f"Failed to write metadata or label map file: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare classification dataset metadata by referencing original images from a nested structure. No images are copied.")
    parser.add_argument('--source_dir', type=str, required=True,
                        help="Root directory of the original dataset (e.g., E:/_MetaFood3D_new_RGBD_videos/RGBD_videos). Expected structure: source_dir/<ClassName>/<InstanceName>/original/<image_files>")
    parser.add_argument('--output_metadata_dir', type=str, required=True,
                        help="Directory where 'metadata.json' will be saved (e.g., data/MetaFood3D_RGBD_classification). This path should be relative to the project root or an absolute path.")
    parser.add_argument('--split_ratio_for_log', type=float, default=0.2,
                        help="Validation split ratio for logging purposes only (default: 0.2). Actual data splitting for TF Datasets is handled by data.py.")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    resolved_output_metadata_dir = Path(args.output_metadata_dir)
    if not resolved_output_metadata_dir.is_absolute():
        resolved_output_metadata_dir = project_root / args.output_metadata_dir

    create_dataset_metadata(args.source_dir, str(resolved_output_metadata_dir), args.split_ratio_for_log)
