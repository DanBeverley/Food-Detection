import numpy as np
import os
import logging
import yaml
import json
import time 
import cv2
from pathlib import Path
from ultralytics import YOLO

try:
    from volume_helpers.density_lookup import lookup_nutritional_info
    from volume_helpers.volume_estimator import estimate_volume_from_depth
except ImportError as e:
    logging.error(f"Failed to import helper functions: {e}")
    # Define dummy functions to prevent immediate crash, though execution will fail later
    def lookup_nutritional_info(*args, **kwargs): return None 
    def estimate_volume_from_depth(*args, **kwargs): return 0.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def _get_project_root() -> Path:
    return Path(__file__).parent

def load_pipeline_config(config_path: str) -> dict:
    if not os.path.isabs(config_path):
        config_path = os.path.join(_get_project_root(), config_path)
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error loading config {config_path}: {e}")
        return {}

class FoodDetector:
    """
    Wrapper for YOLOv8-seg model to handle inference and result parsing.
    """
    def __init__(self, model_path, confidence_threshold=0.5):
        self.model_path = model_path
        self.conf_thresh = confidence_threshold
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            logging.info(f"Loading YOLO model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            logging.info("YOLO model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load YOLO model: {e}")
            self.model = None

    def detect(self, image_path):
        """
        Run inference on an image.
        Returns a list of dicts: {'class_name': str, 'conf': float, 'mask': np.ndarray, 'box': list}
        """
        if self.model is None:
            logging.error("Model not loaded. Cannot run detection.")
            return []

        # Run inference
        results = self.model(image_path, conf=self.conf_thresh, verbose=False)
        
        detections = []
        if not results:
            return detections
            
        result = results[0] # Single image inference
        
        if result.masks is None:
            logging.info("No masks detected.")
            return detections

        # Iterate through detections
        for i, mask_obj in enumerate(result.masks.data):
            # Mask is initially on GPU or float tensor, need to convert to numpy binary mask
            # mask_obj is (H, W) tensor
            mask_cpu = mask_obj.cpu().numpy()
            
            # Resize mask to original image size if needed (YOLO often runs on 640x640)
            # result.orig_shape is (H, W)
            orig_h, orig_w = result.orig_shape
            
            # mask_cpu is likely 640x640, need to resize to orig_h, orig_w
            # Use OpenCV for resizing
            mask_resized = cv2.resize(mask_cpu, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            
            # Binarize
            binary_mask = (mask_resized > 0.5).astype(np.uint8)
            
            # Get Class ID and Name
            cls_id = int(result.boxes.cls[i].item())
            class_name = result.names[cls_id]
            confidence = float(result.boxes.conf[i].item())
            
            detections.append({
                'class_name': class_name,
                'confidence': confidence,
                'mask': binary_mask,
                'box': result.boxes.xyxy[i].tolist() # [x1, y1, x2, y2]
            })
            
        return detections

def analyze_food_item(
    image_path: str,
    config: dict, 
    depth_map_path: str | None = None,  
    point_cloud_path: str | None = None, 
    mesh_file_path: str | None = None,  
    output_dir: str | None = None, 
    save_steps: bool = False, 
    display_results: bool = True,
    known_food_class: str | None = None, 
    usda_api_key: str | None = None, 
    mask_path: str | None = None,
    volume_estimation_method: str = 'depth', 
    camera_intrinsics_key: str = 'default',
    custom_camera_intrinsics: dict | None = None,
    volume_estimation_config: dict | None = None
) -> dict | None:
    """
    Main pipeline for Multi-Food Analysis using YOLOv8-seg.
    """
    start_time = time.time()
    results = {
        'food_items': [],
        'total_summary': {
            'total_calories': 0.0,
            'total_mass_g': 0.0,
            'item_count': 0
        },
        'timing': {},
        'error_messages': []
    }
    
    image_basename = os.path.basename(image_path)
    logging.info(f"Analyzing Image: {image_basename} (Multi-Object Mode)")

    # 1. Load Depth Map (Shared across all objects)
    t0_depth = time.time()
    depth_map = None
    if depth_map_path and os.path.exists(depth_map_path):
        if depth_map_path.endswith('.npy'):
            depth_map = np.load(depth_map_path)
        else:
            # Assuming depth map image (grayscale)
            depth_map = cv2.imread(depth_map_path, cv2.IMREAD_ANYDEPTH)
    
    if depth_map is None:
        logging.warning("Depth map not found or failed to load. Volume estimation will be 0.")
        # Minimal dummy depth for testing pipeline flow if needed, or just handle None later
        # depth_map = np.ones((480, 640)) * 1000 
    results['timing']['load_depth'] = time.time() - t0_depth

    # 2. Run Segmentation & Detection (YOLO)
    t0_inf = time.time()
    
    # Locate Model
    # Try config first, then default fallback
    yolo_model_path = config.get('models', {}).get('segmentation_yolo', 'yolov8n-seg.pt')
    if not os.path.exists(yolo_model_path):
        # Fallback check commonly used paths
        potential_paths = [
            'yolov8n-seg.pt',
            'models/segmentation/yolov8n-seg.pt',
            'runs/segment/food_seg_n/weights/best.pt'
        ]
        found = False
        for p in potential_paths:
            if os.path.exists(p):
                yolo_model_path = p
                found = True
                break
        if not found:
            logging.error(f"YOLO model not found at {yolo_model_path} or default locations.")
            results['error_messages'].append("YOLO model not found")
            return results

    detector = FoodDetector(yolo_model_path, confidence_threshold=0.3)
    detections = detector.detect(image_path)
    results['timing']['inference'] = time.time() - t0_inf
    
    logging.info(f"Detected {len(detections)} food items.")

    # 3. Process Each Detected Item
    for idx, item in enumerate(detections):
        class_name = item['class_name']
        confidence = item['confidence']
        mask = item['mask']
        
        logging.info(f"Processing Item {idx+1}: {class_name} ({confidence:.2f})")
        
        item_result = {
            'id': idx + 1,
            'label': class_name,
            'confidence': confidence,
            'volume_cm3': 0.0,
            'density_g_cm3': None,
            'mass_g': None,
            'calories_kcal': None,
            'warnings': []
        }

        # A. Volume Estimation
        if depth_map is not None:
            try:
                # Ensure mask and depth shapes match
                if mask.shape != depth_map.shape:
                    logging.warning(f"Shape mismatch: Mask {mask.shape} vs Depth {depth_map.shape}. Resizing mask.")
                    mask = cv2.resize(mask, (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_NEAREST)

                vol = estimate_volume_from_depth(
                    depth_map=depth_map,
                    segmentation_mask=mask,
                    camera_intrinsics_key=camera_intrinsics_key,
                    custom_intrinsics=custom_camera_intrinsics,
                    config=volume_estimation_config
                )
                item_result['volume_cm3'] = vol
            except Exception as e:
                logging.error(f"Volume estimation failed for {class_name}: {e}")
                item_result['warnings'].append(f"Volume error: {e}")

        # B. Nutrition Lookup
        if item_result['volume_cm3'] > 0:
            info = lookup_nutritional_info(class_name, usda_api_key)
            if info:
                density = info.get('density') # g/cm3
                cals_per_100g = info.get('calories_kcal_per_100g')
                
                item_result['density_g_cm3'] = density
                
                if density:
                    mass = item_result['volume_cm3'] * density
                    item_result['mass_g'] = mass
                    
                    if cals_per_100g is not None:
                        total_cals = (mass / 100.0) * cals_per_100g
                        item_result['calories_kcal'] = total_cals
                        
                        # Accumulate Totals
                        results['total_summary']['total_calories'] += total_cals
                        results['total_summary']['total_mass_g'] += mass
                else:
                    item_result['warnings'].append("Density not found in DB")
            else:
                item_result['warnings'].append("Nutrition info not found")
        
        results['food_items'].append(item_result)
        results['total_summary']['item_count'] += 1
        
        # Save Mask Step
        if save_steps and output_dir:
            mask_filename = os.path.join(output_dir, f"{os.path.splitext(image_basename)[0]}_item{idx}_{class_name}_mask.png")
            cv2.imwrite(mask_filename, mask * 255)

    results['timing']['total'] = time.time() - start_time
    
    # 4. Final Logging
    logging.info("--- ANALYSIS SUMMARY ---")
    logging.info(f"Total Items: {results['total_summary']['item_count']}")
    logging.info(f"Total Calories: {results['total_summary']['total_calories']:.2f} kcal")
    logging.info(f"Total Mass: {results['total_summary']['total_mass_g']:.2f} g")
    
    if display_results:
        print("\n=== FOOD ANALYSIS REPORT ===")
        print(f"Image: {image_basename}")
        print(f"Items Detected: {len(results['food_items'])}")
        for item in results['food_items']:
            print(f"  - {item['label']} ({item['confidence']:.2f})")
            print(f"    Volume: {item['volume_cm3']:.2f} cm3")
            if item['calories_kcal']:
                print(f"    Calories: {item['calories_kcal']:.2f} kcal")
            else:
                print(f"    Calories: N/A")
        print(f"TOTAL CALORIES: {results['total_summary']['total_calories']:.2f} kcal")
        print("============================")

    return results

if __name__ == "__main__":
    # Simple CLI for testing
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--depth', default=None)
    parser.add_argument('--model', default='yolov8n-seg.pt')
    parser.add_argument('--api_key', default=None)
    args = parser.parse_args()
    
    # Mock config
    cfg = {'models': {'segmentation_yolo': args.model}}
    
    analyze_food_item(args.image, cfg, args.depth, usda_api_key=args.api_key)