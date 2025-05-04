import argparse
import os
import yaml
import numpy as np
import tensorflow as tf
from PIL import Image
import logging
from pathlib import Path
import cv2 

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Utilities
def get_project_root()->Path:
    return Path(__file__).parent.parent.parent

def load_and_preprocess_image(image_path:str, target_size_hw:tuple) -> tuple[np.ndarray, tuple]:
    """
    Loads and preprocesses an image for TFLite inference.

    Args:
        image_path: Path to the input image.
        target_size_hw: Target size as (height, width).

    Returns:
        A tuple containing:
         - Preprocessed image as a numpy array (batch dim added, normalized).
         - Original image dimensions (height, width).
    """
    try:
        img = Image.open(image_path).convert("RGB")
        original_size_wh = img.size # (width, height)
        original_size_hw = (original_size_wh[1], original_size_wh[0]) 
        # Resize to model input size
        target_size_wh = (target_size_hw[1], target_size_hw[0])
        img_resized = img.resize(target_size_wh, Image.BILINEAR)
        img_array = np.array(img_resized, dtype = np.float32)
        img_array = img_array/255.0
        # Add batch dimension
        input_data = np.expand_dims(img_array, axis = 0)
        return input_data, original_size_hw
    except FileNotFoundError:
        logging.error(f"Image file not found: {image_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading or preprocessing image {image_path}: {e}")
        raise

def postprocess_mask(raw_mask:np.ndarray, target_size_hw:tuple, num_classes:int, threshold:float=0.5) -> np.ndarray:
    """
    Postprocesses the raw output mask from the TFLite model.

    Args:
        raw_mask: The output tensor from the interpreter.
        target_size_hw: The target image dimensions (height, width) to resize back to.
        num_classes: The number of classes the model was trained for.
        threshold: Confidence threshold for binary segmentation (if num_classes <= 2).

    Returns:
        Processed mask (resized, thresholded/argmaxed) as a numpy array (H, W).
    """
    # Remove batch dimension if exist
    if raw_mask.ndim > 3:
        raw_mask = np.squeeze(raw_mask, axis=0) # (b, h, w, c) -> (h, w, c)
    model_output_size_hw = raw_mask.shape[:2]
    # Process based on number of classes
    if num_classes == 2 or raw_mask.shape[-1] == 1:
        # Binary classification (single channel output) - Apply threshold
        processed_mask = (raw_mask > threshold).astype(np.uint8)
        if processed_mask.ndim > 2 and processed_mask.shape[-1] == 1:
            processed_mask = np.squeeze(processed_mask, axis =- 1) # (h, w)
    elif num_classes > 2:
        # Multi-class classification - argmax
        processed_mask = np.argmax(raw_mask, axis=-1).astype(np.uint8) # (h, w)
    else:
        logging.error(f"Invalid num_classes ({num_classes}) for mask postprocessing")
        # Default to thresholding single channel if available
        processed_mask = (raw_mask > threshold).astype(np.uint8)
        if processed_mask.ndim > 2 and processed_mask.shape[-1] == 1:
            processed_mask = np.squeeze(processed_mask, axis =-1)
    # Resize mask back to target size using NEAREST
    # OpenCV resize takes (w, h)
    target_size_wh = (target_size_hw[1], target_size_hw[0])
    final_mask = cv2.resize(processed_mask, target_size_wh, interpolation=cv2.INTER_NEAREST)
    return final_mask #(target_h, target_w)

def overlay_mask_on_image(image_path:str, mask:np.ndarray, alpha:float=0.5) -> np.ndarray|None:
    """Overlays the predicted mask on the original image."""
    try:
        original_image = cv2.imread(image_path)
        if original_image is None:
            logging.error(f"Failed to read original image for overlay: {image_path}")
            return None
        # Ensure mask and image has the same (h, w)
        if original_image.shape[:2] != mask.shape[:2]:
            logging.warning(f"Original image size {original_image.shape[:2]} and mask size {masks.shape[:2]} differ. Resizing mask for overlay")
            mask = cv2.resize(mask.astype(np.uint8), original_image.shape[1], original_image.shape[0], interpolation=cv2.INTER_NEAREST)
        # Ensure mask is single channel 0 or 1 for simple overlay
        if mask.max() > 1: # If mask has class indices > 1, use non-zero as mask
            binary_mask = (mask>0).astype(np.uint8)
        else:
            binary_mask = mask.astype(np.uint8)
        # Create colored mask (e.g., green)
        colored_mask = np.zeros_like(original_image)
        colored_mask[binary_mask == 1] = [0, 255, 0] # Green color for mask
        # Blend images
        overlayed_image = cv2.addWeighted(original_image, 1, colored_mask, alpha, 0)
        overlayed_image = cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RBG)
        return overlayed_image
    except Exception as e:
        logging.error(f"Error overlaying mask: {e}")
        return None

# Core logic

def load_segmentation_model(tflite_model_path:str):
    """Loads the TFLite segmentation model interpreter"""
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        logging.info(f"TFLite segmentation model loaded: {tflite_model_path}")
        logging.debug(f"Input details: {input_details}")
        logging.debug(f"Output details: {output_details}")
        return interpreter, input_details, output_details
    except Exception as e:
        logging.error(f"Failed to load TFLite interpreter from {tflite_model_path}: {e}")
        raise

def run_segmentation_inference(
    interpreter:tf.lite.Interpreter,
    input_details:list,
    output_details:list,
    image_path:str,
    model_input_size_hw:tuple,
    output_resize_shape_hw:tuple, # e.g., original image shape or depth map shape
    num_classes:int = 2) -> np.ndarray|None:
    """
    Runs segmentation inference on a single image.

    Args:
        interpreter: Loaded TFLite interpreter.
        input_details: Interpreter input details.
        output_details: Interpreter output details.
        image_path: Path to the input image.
        model_input_size_hw: Expected input size (H, W) of the TFLite model.
        output_resize_shape_hw: Target shape (H, W) to resize the final mask to.
        num_classes: Number of output classes for the model.

    Returns:
        Processed segmentation mask (resized to output_resize_shape_hw) or None on failure.
    """
    try:
        # Preprocess image
        input_data, _ = load_and_preprocess_image(image_path, model_input_size_hw)
        # Check input tensor type and scale if necessary (e.g. for UINT8 models)
        input_dtype = input_details[0]["dtype"]
        if input_dtype == np.uint8:
            scale, zero_point = input_details[0]["quantization"]
            input_data = (input_data/scale+zero_point).astype(input_dtypes)
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        raw_mask = interpreter.get_tensor(output_details[0]["index"])
        # Handle quantized output 
        output_dtype = output_details[0]["dtype"]
        if output_dtypes == np.uint8:
            scale, zero_point = output_details[0]["quantization"]
            # Dequantize - check scale and zero point are valid
            if scale != 0:
                raw_mask = (raw_mask.astype(np.float32)-zero_points)*scale
            else:
                logging.warning("Output tensor is quantized (UINT8) but scale is zero. Using raw values")
                raw_mask = raw_mask.astype(np.float32)
        logging.debug("Inference completed")
        # Postprocess output mask
        final_mask = postprocess_mask(raw_mask, output_resize_shape_hw, num_classes)
        logging.debug(f"Mask postprocessed to size: {final_mask.shape}")
        # Mask values should be class indices (0, 1, ... N - 1)
        logging.debug(f"Unique values in final mask: {np.unique(final_mask)}")
        return final_mask
    except Exception as e:
        logging.exception(f"Error during segmentation inference for {image_path}: {e}")
        return None

def predict_standalone(config_path:str, image_path:str, output_path:str = None, show_overlay:bool=False):
    """Loads TFLite model and performs segmentation prediction (Standalone mode)."""
    try:
        project_root = _get_project_root()
        if not os.path.isabs(config_path):
            config_path = os.path.join(project_root, config_path)
        with open(config_path, "r")as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        return
    except Exception as e:
        logging.error(f"Error reading or parsing configuration file {config_path}: {e}")
        return
    try:
        tflite_model_rel_path = config["models"]["segmentation_tflite"]
        model_input_size_hw = tuple(config["model_params"]["segmentation_input_size"])
        num_classes = config.get("model", {}).get("num_classes", 2)
    except KeyError as e:
        logging.error(f"Missing key in configuration file {config_path}: {e}")
        return
    tflite_model_path = os.path.join(project_root, tflite_model_rel_path)
    if not os.path.exists(tflite_model_path):
        logging.error(f"TFLite model file not found: {tflite_model_path}")
        return 
    # Load model
    try:
        interpreter, input_details, output_details = load_segmentation_model(tflite_model_path)
    except Exception:
        return 
    # Get original size to resize mask
    try:
        with Image.open(image_path) as img:
            original_size_hw = (img.height, img.width)
    except Exception as e:
        logging.error(f"Failed to get original image size from {image_path}: {e}")
        return 
    
    # Run inference
    final_mask = run_segmentation_inference(interpreter, input_details, output_details, 
                                            image_path, model_input_size_hw, original_size_hw,
                                            num_classes)
    if final_mask is None:
        logging.error("Segmentation inference faied")
        return 
    
    # Handle output
    if output_path:
        try:
            # Save mask as an image
            save_mask = final_mask.astype(np.uint8)
            if num_classes > 1: # Scale to make class visible in grayscale
                scale_factor = 255//(num_classes - 1 if num_classes > 1 else 1)
                save_mask = (save_mask * scale_factor).astype(np.uint8)
            else: # Binary mask 0 or 1 -> 0 or 255
                save_mask = (save_mask * 255).astype(np.uint8)
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            cv2.imwrite(output_path, save_mask)
            logging.info(f"Segmentation mask saved to: {output_path}")
        except Exception as e:
            logging.error(f"Failed to save output mask to {output_path}: {e}")
    if show_overlay:
        overlayed_image = overlay_mask_on_image(image_path, final_mask)
        if overlayed_image is not None:
             # Display using matplotlib or OpenCV
             try:
                 import matplotlib.pyplot as plt
                 plt.imshow(overlayed_image) # Already RGB
                 plt.title(f"Segmentation Overlay ({os.path.basename(image_path)})")
                 plt.axis('off')
                 plt.show()
             except ImportError:
                 logging.warning("Matplotlib not installed. Cannot show overlay inline. Try saving the output.")
                 # Fallback: Save overlay image if output path was also given
                 if output_path:
                     overlay_output_path = os.path.splitext(output_path)[0] + "_overlay.png"
                     try:
                         # Convert RGB to BGR for cv2.imwrite
                         cv2.imwrite(overlay_output_path, cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR))
                         logging.info(f"Overlay image saved to: {overlay_output_path}")
                     except Exception as e:
                         logging.error(f"Failed to save overlay image: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run segmentation prediction using a TFLite model (Standalone).")
    # Point to the central pipeline config now
    parser.add_argument('--config', type=str, default='config_pipeline.yaml',
                        help="Path to the pipeline YAML configuration file.")
    parser.add_argument('--image', type=str, required=True,
                        help="Path to the input image.")
    parser.add_argument('--output', type=str, default=None,
                        help="Path to save the output mask image.")
    parser.add_argument('--show', action='store_true',
                        help="Show the original image with the mask overlayed (requires matplotlib).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Debug logging enabled.")

    predict_standalone(args.config, args.image, args.output, args.show) 