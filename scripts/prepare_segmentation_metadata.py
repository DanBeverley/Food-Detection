import os
import json
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration --- 
# Modify these if your structure differs
ORIGINAL_IMAGE_FOLDER_NAME = "original"  
SEGMENTATION_MASK_FOLDER_NAME = "masks" 
# Set to True if mask name is image_name + "_mask" + extension
# Set to False if mask name is exactly the same as image name
MASK_HAS_SUFFIX = False 
MASK_SUFFIX = "_mask" 
# -------------------

def create_segmentation_metadata(source_rgbd_base_dir_path: str, output_metadata_dir_path: str):
    """
    Scans a source directory for image-mask pairs based on a nested structure
    (Class/Instance/original and Class/Instance/segmentation) and generates a 
    metadata.json file containing absolute paths to these pairs.
    No files are copied.

    Expected structure:
    - source_rgbd_base_dir_path/<ClassName>/<InstanceName>/original/<image_file>
    - source_rgbd_base_dir_path/<ClassName>/<InstanceName>/segmentation/<mask_file>

    Args:
        source_rgbd_base_dir_path (str): Root directory of the original dataset 
                                         (e.g., 'E:\\_MetaFood3D_new_RGBD_videos\\RGBD_videos').
        output_metadata_dir_path (str): Directory where 'metadata.json' will be created
                                       (e.g., 'data/MetaFood3D_RGBD_segmentation').
    """
    source_dir = Path(source_rgbd_base_dir_path)
    output_meta_dir = Path(output_metadata_dir_path)
    metadata_file_path = output_meta_dir / "metadata.json"

    if not source_dir.is_dir():
        logging.error(f"Source directory not found: {source_dir}")
        return

    os.makedirs(output_meta_dir, exist_ok=True)

    all_metadata_pairs = []
    total_pairs_found = 0
    total_images_scanned = 0
    missing_mask_count = 0

    logging.info(f"Scanning source directory: {source_dir} to generate segmentation metadata.")
    class_folders = [d for d in source_dir.iterdir() if d.is_dir()]
    if not class_folders:
        logging.error(f"No class subdirectories found in {source_dir}")
        return

    logging.info(f"Found {len(class_folders)} potential class folders.")

    for class_folder in class_folders:
        class_name = class_folder.name
        instance_folders = [d for d in class_folder.iterdir() if d.is_dir()]
        if not instance_folders:
            logging.warning(f"No instance subdirectories found for class '{class_name}'.")
            continue

        logging.debug(f"Processing class '{class_name}' with {len(instance_folders)} instances...")

        for instance_folder in instance_folders:
            instance_name = instance_folder.name
            original_folder = instance_folder / ORIGINAL_IMAGE_FOLDER_NAME
            mask_folder = instance_folder / SEGMENTATION_MASK_FOLDER_NAME

            if not original_folder.is_dir():
                logging.warning(f"  Class '{class_name}', Instance '{instance_name}': Missing '{ORIGINAL_IMAGE_FOLDER_NAME}' folder. Skipping instance.")
                continue
            if not mask_folder.is_dir():
                logging.warning(f"  Class '{class_name}', Instance '{instance_name}': Missing '{SEGMENTATION_MASK_FOLDER_NAME}' folder. Skipping instance.")
                continue

            image_files = list(original_folder.glob('*[.jpg][.jpeg][.png][.gif]'))
            if not image_files:
                continue

            for img_path in image_files:
                total_images_scanned += 1
                img_stem = img_path.stem
                img_ext = img_path.suffix
                
                if MASK_HAS_SUFFIX:
                    expected_mask_stem = img_stem + MASK_SUFFIX
                else:
                    expected_mask_stem = img_stem
                
                # Attempt to find mask with potentially different extensions (e.g., .png mask for .jpg image)
                found_mask_path = None
                for mask_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']:
                    potential_mask_path = mask_folder / (expected_mask_stem + mask_ext)
                    if potential_mask_path.is_file():
                        found_mask_path = potential_mask_path
                        break 
                
                if found_mask_path:
                    try:
                        pair_entry = {
                            "image_path": str(img_path.resolve()),
                            "mask_path": str(found_mask_path.resolve())
                        }
                        all_metadata_pairs.append(pair_entry)
                        total_pairs_found += 1
                    except Exception as e:
                        logging.error(f"Error processing pair for image {img_path}: {e}")
                else:
                    missing_mask_count += 1
                    logging.warning(f"  Missing corresponding mask for image: {img_path} (looked for stem '{expected_mask_stem}' in {mask_folder})")

    if not all_metadata_pairs:
        logging.error("No image-mask pairs were found. Metadata file will not be created.")
        return

    try:
        with open(metadata_file_path, 'w') as f:
            json.dump(all_metadata_pairs, f, indent=2)
        logging.info(f"Successfully created segmentation metadata file at: {metadata_file_path}")
        logging.info(f"Total image files scanned: {total_images_scanned}")
        logging.info(f"Total image-mask pairs found: {total_pairs_found}")
        if missing_mask_count > 0:
            logging.warning(f"Could not find matching masks for {missing_mask_count} images.")

    except IOError as e:
        logging.error(f"Failed to write metadata file: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare segmentation dataset metadata by finding image-mask pairs in a nested structure. No files are copied.")
    parser.add_argument('--source_dir', type=str, required=True,
                        help=f"Root directory of the original dataset (e.g., E:/_MetaFood3D_new_RGBD_videos/RGBD_videos). Expected structure: source_dir/<ClassName>/<InstanceName>/{ORIGINAL_IMAGE_FOLDER_NAME}/<image_files> and corresponding masks in .../{SEGMENTATION_MASK_FOLDER_NAME}/")
    parser.add_argument('--output_metadata_dir', type=str, required=True,
                        help="Directory where 'metadata.json' will be saved (e.g., data/MetaFood3D_RGBD_segmentation). Relative paths are resolved from project root.")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    resolved_output_metadata_dir = Path(args.output_metadata_dir)
    if not resolved_output_metadata_dir.is_absolute():
        resolved_output_metadata_dir = project_root / args.output_metadata_dir

    create_segmentation_metadata(args.source_dir, str(resolved_output_metadata_dir))
