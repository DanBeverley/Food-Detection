import os
import json
import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ORIGINAL_IMAGE_FOLDER_NAME = "original"  
SEGMENTATION_MASK_FOLDER_NAME = "masks" 
DEPTH_MAP_FOLDER_NAME = "depth"  

DEPTH_MAP_EXTENSIONS = ['.jpg'] 
POINT_CLOUD_EXTENSIONS = ['.ply'] 

MASK_HAS_SUFFIX = False 
MASK_SUFFIX = "_mask"

def _load_and_process_point_cloud(pc_path_str: str, num_points_target: int, normalization_method_str: str) -> np.ndarray:
    """
    Loads a point cloud from a file path, processes it, and returns a NumPy array.
    This is the pure Python version for preprocessing.
    """
    try:
        import trimesh
        
        if not pc_path_str or not os.path.exists(pc_path_str):
            return np.zeros((num_points_target, 3), dtype=np.float32)

        mesh_or_points = trimesh.load(pc_path_str, process=False) 

        if isinstance(mesh_or_points, trimesh.Trimesh):
            points = mesh_or_points.vertices
        elif isinstance(mesh_or_points, trimesh.points.PointCloud):
            points = mesh_or_points.vertices
        else:
            return np.zeros((num_points_target, 3), dtype=np.float32)

        if points.shape[0] == 0:
            return np.zeros((num_points_target, 3), dtype=np.float32)

        points = points.astype(np.float32)
        
        current_num_points = points.shape[0]
        if current_num_points > num_points_target:
            indices = np.random.choice(current_num_points, num_points_target, replace=False)
            points = points[indices]
        elif current_num_points < num_points_target:
            if current_num_points == 0:
                return np.zeros((num_points_target, 3), dtype=np.float32)
            padding_indices = np.random.choice(current_num_points, num_points_target - current_num_points, replace=True)
            points = np.vstack((points, points[padding_indices]))
        
        if normalization_method_str != 'none':
            points_mean = np.mean(points, axis=0)
            points_centered = points - points_mean
            if normalization_method_str == 'unit_sphere':
                max_dist = np.max(np.linalg.norm(points_centered, axis=1))
                points_normalized = points_centered / (max_dist + 1e-6)
            elif normalization_method_str == 'unit_cube':
                max_abs_coord = np.max(np.abs(points_centered))
                points_normalized = points_centered / (max_abs_coord + 1e-6)
            elif normalization_method_str == 'centered_only':
                points_normalized = points_centered
            else:
                points_normalized = points_centered
            points = points_normalized
        
        return points.astype(np.float32)

    except Exception as e:
        logging.error(f"Error processing point cloud {pc_path_str}: {e}")
        return np.zeros((num_points_target, 3), dtype=np.float32)

 

def create_segmentation_metadata(source_rgbd_base_dir_path: str, 
                                 output_metadata_dir_path: str, 
                                 source_point_cloud_base_dir_path: str = None,
                                 num_points_target: int = 4096,
                                 pc_normalization: str = 'unit_sphere'):
    """
    Scans a source directory for image-mask-depth-pointcloud tuples based on a nested structure
    and generates a metadata.json file containing absolute paths to these items.
    No files are copied.

    Expected structure for RGB-D data:
    - source_rgbd_base_dir_path/<ClassName>/<InstanceName>/original/<image_file>
    - source_rgbd_base_dir_path/<ClassName>/<InstanceName>/masks/<mask_file>
    - source_rgbd_base_dir_path/<ClassName>/<InstanceName>/depth/<depth_map_file> (e.g., image_stem.png)
    
    Expected structure for Point Cloud data:
    - <source_point_cloud_base_dir_path>/<ClassName>/<InstanceName>/<DerivedPCFileName>.ply
      (Example: <source_point_cloud_base_dir_path>/Almond(bowl)/almond_1/Almond_1_sampled_1.ply for image 0.jpg)

    Args:
        source_rgbd_base_dir_path (str): Root directory of the original RGB, Mask, Depth dataset 
                                         (e.g., 'E:\\_MetaFood3D_new_RGBD_videos\\RGBD_videos').
        output_metadata_dir_path (str): Directory where 'metadata.json' will be created
                                       (e.g., 'data/MetaFood3D_RGBD_segmentation').
        source_point_cloud_base_dir_path (str): Root directory for Point Cloud files
                                                (e.g., 'E:\\_MetaFood3D_new_Point_cloud\\Point_cloud\\4096').
    """
    source_dir = Path(source_rgbd_base_dir_path)
    output_meta_dir = Path(output_metadata_dir_path)
    source_pc_base_dir = Path(source_point_cloud_base_dir_path) if source_point_cloud_base_dir_path else None
    metadata_file_path = output_meta_dir / "metadata.json"
    
    # Create directory for preprocessed point clouds
    preprocessed_pc_dir = output_meta_dir / "preprocessed_point_clouds"
    preprocessed_pc_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Pre-processed point clouds will be saved to: {preprocessed_pc_dir}")

    if not source_dir.is_dir():
        logging.error(f"Source directory not found: {source_dir}")
        return
    
    if source_pc_base_dir and not source_pc_base_dir.is_dir():
        logging.warning(f"Point cloud directory not found: {source_pc_base_dir}. Point cloud paths will be set to null.")
        source_pc_base_dir = None

    os.makedirs(output_meta_dir, exist_ok=True)

    all_metadata_entries = []
    total_entries_found = 0
    total_images_scanned = 0
    missing_mask_count = 0
    missing_depth_count = 0 
    missing_point_cloud_count = 0 

    logging.info(f"Scanning source directory: {source_dir} to generate full segmentation metadata.")
    logging.info(f"Point cloud source directory: {source_pc_base_dir}")
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
            depth_folder = instance_folder / DEPTH_MAP_FOLDER_NAME 
            # Handle optional point cloud directory
            current_pc_instance_dir = source_pc_base_dir / class_name / instance_name if source_pc_base_dir else None

            if not original_folder.is_dir():
                logging.warning(f"  Class '{class_name}', Instance '{instance_name}': Missing '{ORIGINAL_IMAGE_FOLDER_NAME}' folder. Skipping instance.")
                continue
            
            instance_point_cloud_npy_path_str = None
            if current_pc_instance_dir and current_pc_instance_dir.is_dir():
                found_pc_file_for_instance = False
                for item in current_pc_instance_dir.iterdir():
                    if item.is_file() and item.name.lower().endswith("_sampled_1.ply"):
                        original_ply_path_str = str(item.resolve())
                        logging.debug(f"  Found point cloud for instance '{instance_name}' (file: {item.name}): {original_ply_path_str}")
                        
                        # Process the point cloud
                        processed_points = _load_and_process_point_cloud(
                            original_ply_path_str,
                            num_points_target,
                            pc_normalization
                        )
                        
                        # Save the result as a .npy file
                        npy_filename = f"{class_name}_{instance_name}_{item.stem}.npy"
                        npy_save_path = preprocessed_pc_dir / npy_filename
                        np.save(npy_save_path, processed_points)
                        
                        # Store the absolute path for now, will convert to relative later
                        instance_point_cloud_npy_path_str = str(npy_save_path.resolve())
                        logging.debug(f"  Processed and saved point cloud to: {npy_save_path}")
                        
                        found_pc_file_for_instance = True
                        break # Found the unique _sampled_1.ply for this instance
                
                if not found_pc_file_for_instance:
                    # Log if no file ending with _sampled_1.ply was found in the directory
                    logging.warning(f"  Point cloud file ending with '_sampled_1.ply' not found in {current_pc_instance_dir} for instance '{instance_name}'. Point clouds will be null for this instance.")
            elif current_pc_instance_dir:
                logging.warning(f"  Point cloud data directory not found at '{current_pc_instance_dir}'. Point clouds will be null for instance '{instance_name}'.")
            else:
                logging.debug(f"  Point cloud processing disabled. Point clouds will be null for instance '{instance_name}'.")

            # Masks, depth folders check 
            if not mask_folder.is_dir():
                logging.warning(f"  Class '{class_name}', Instance '{instance_name}': Missing '{SEGMENTATION_MASK_FOLDER_NAME}' folder. Masks will be null for this instance.")
            if not depth_folder.is_dir(): 
                logging.warning(f"  Class '{class_name}', Instance '{instance_name}': Missing '{DEPTH_MAP_FOLDER_NAME}' folder. Depth maps will be null for this instance.")

            image_files = list(original_folder.glob('*[.jpg][.jpeg][.png][.gif]'))
            if not image_files:
                continue

            for img_path in image_files:
                total_images_scanned += 1
                img_stem = img_path.stem
                
                # Find Mask 
                found_mask_path_str = None
                if mask_folder.is_dir():
                    expected_mask_stem = img_stem + MASK_SUFFIX if MASK_HAS_SUFFIX else img_stem
                    for mask_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']:
                        potential_mask_path = mask_folder / (expected_mask_stem + mask_ext)
                        if potential_mask_path.is_file():
                            found_mask_path_str = str(potential_mask_path.resolve())
                            break 
                if not found_mask_path_str:
                    missing_mask_count += 1
                    logging.debug(f"  Missing corresponding mask for image: {img_path} (looked for stem '{expected_mask_stem}' in {mask_folder})")

                # Find Depth Map 
                found_depth_path_str = None
                if depth_folder.is_dir():
                    for depth_ext in DEPTH_MAP_EXTENSIONS:
                        potential_depth_path = depth_folder / (img_stem + depth_ext)
                        if potential_depth_path.is_file():
                            found_depth_path_str = str(potential_depth_path.resolve())
                            break
                if not found_depth_path_str:
                    missing_depth_count += 1
                    logging.debug(f"  Missing corresponding depth map for image: {img_path} (looked for stem '{img_stem}' in {depth_folder})")

                # Point cloud path is now taken from the instance-level find 
                # No per-image search needed for point cloud 
                if not instance_point_cloud_npy_path_str and current_pc_instance_dir and not current_pc_instance_dir.is_dir():
                    missing_point_cloud_count +=1 
                elif not instance_point_cloud_npy_path_str and current_pc_instance_dir and current_pc_instance_dir.is_dir():
                    missing_point_cloud_count +=1 

                try:
                    entry = {
                        "image_path": str(img_path.resolve()),
                        "mask_path": str(Path(found_mask_path_str).resolve()) if found_mask_path_str else None,
                        "depth_map_path": str(Path(found_depth_path_str).resolve()) if found_depth_path_str else None,
                        "point_cloud_path": str(Path(instance_point_cloud_npy_path_str).resolve()) if instance_point_cloud_npy_path_str else None,
                        "class_name": class_name,
                        "instance_name": instance_name
                    }
                    all_metadata_entries.append(entry)
                    total_entries_found += 1
                except Exception as e:
                    logging.error(f"Error processing entry for image {img_path}: {e}")

    if not all_metadata_entries:
        logging.error("No metadata entries were compiled. Metadata file will not be created.")
        return

    try:
        with open(metadata_file_path, 'w') as f:
            json.dump(all_metadata_entries, f, indent=2)
        logging.info(f"Successfully created segmentation metadata file at: {metadata_file_path}")
        logging.info(f"Total image files scanned: {total_images_scanned}")
        logging.info(f"Total metadata entries created: {total_entries_found}")
        if missing_mask_count > 0:
            logging.warning(f"Could not find matching masks for {missing_mask_count} images (path set to null).")
        if missing_depth_count > 0: 
            logging.warning(f"Could not find matching depth maps for {missing_depth_count} images (path set to null).")
        if missing_point_cloud_count > 0: 
            logging.warning(f"Could not find matching point clouds for {missing_point_cloud_count} images (path set to null).")


    except IOError as e:
        logging.error(f"Failed to write metadata file: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare segmentation dataset metadata by finding image, mask, depth map, and point cloud paths in a nested structure. No files are copied.")
    parser.add_argument('--source_dir', type=str, required=True,
                        help=f"Root directory of the original dataset (e.g., E:/_MetaFood3D_new_RGBD_videos/RGBD_videos). Expected structure: source_dir/<ClassName>/<InstanceName>/{ORIGINAL_IMAGE_FOLDER_NAME}/<image_files>, and corresponding files in .../{SEGMENTATION_MASK_FOLDER_NAME}/, .../{DEPTH_MAP_FOLDER_NAME}/")
    parser.add_argument('--output_metadata_dir', type=str, required=True,
                        help="Directory where 'metadata.json' will be saved (e.g., data/MetaFood3D_RGBD_segmentation). Relative paths are resolved from project root.")
    parser.add_argument('--source_point_cloud_dir', type=str, required=False, default=None,
                        help="Base directory for point cloud files (e.g., E:/_MetaFood3D_new_Point_cloud/Point_cloud/4096). "
                             "Expected structure: <source_point_cloud_dir>/<ClassName>/<InstanceName>/<DerivedPCFileName>.ply. "
                             "Optional - if not provided, point cloud paths will be set to null.")
    parser.add_argument('--num_points', type=int, default=4096,
                        help="Target number of points for point cloud preprocessing (default: 4096)")
    parser.add_argument('--pc_normalization', type=str, default='unit_sphere',
                        choices=['unit_sphere', 'unit_cube', 'centered_only', 'none'],
                        help="Point cloud normalization method (default: unit_sphere)")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    resolved_output_metadata_dir = Path(args.output_metadata_dir)
    if not resolved_output_metadata_dir.is_absolute():
        resolved_output_metadata_dir = project_root / args.output_metadata_dir

    create_segmentation_metadata(args.source_dir, 
                                 str(resolved_output_metadata_dir),
                                 args.source_point_cloud_dir,
                                 args.num_points,
                                 args.pc_normalization)
