import argparse
import os
import yaml
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
import cv2 
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Utilities
def get_project_root()->Path:
    return Path(__file__).parent.parent.parent

def load_and_preprocess_image(image_path:str, target_size_hw:tuple) -> tuple[np.ndarray, tuple]:
    """Loads, resizes, flattens, normalizes, and batches an image."""
    try:
        target_h, target_w = target_size_hw # e.g., (64, 64)

        img = Image.open(image_path).convert("RGB")
        original_size_wh = img.size
        original_size_hw = (original_size_wh[1], original_size_wh[0])
        
        # Resize to target_size_hw for the model
        img_resized = img.resize((target_w, target_h), Image.Resampling.BILINEAR) # PIL resize is (W,H)
        img_array = np.array(img_resized, dtype=np.float32)
        
        # Apply normalization (e.g., EfficientNet's preprocess_input)
        img_array = preprocess_input(img_array) # Expects H,W,C float32
        
        # Add batch dimension: (H, W, C) -> (1, H, W, C)
        input_data = np.expand_dims(img_array, axis=0) # Shape e.g. (1, 128, 128, 3)
        logging.info(f"Image {Path(image_path).name}: Resized to {img_array.shape}, final shape: {input_data.shape}")
        logging.info(f"Image {Path(image_path).name}: Final preprocessed shape for TFLite: {input_data.shape}")
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
    logging.info(f"PREDICT_SEG_DEBUG (postprocess_mask - INPUT raw_mask stats):\n"
                 f"  Shape={raw_mask.shape}, dtype={raw_mask.dtype}\n"
                 f"  Min={np.min(raw_mask):.4f}, Max={np.max(raw_mask):.4f}, Mean={np.mean(raw_mask):.4f}")

    # --- Apply Sigmoid to convert logits to probabilities ---
    # Assuming raw_mask contains logits from the model
    probabilities = 1 / (1 + np.exp(-raw_mask.astype(np.float32))) # Ensure float for exp
    logging.info(f"PREDICT_SEG_DEBUG (postprocess_mask - Probabilities after Sigmoid):\n"
                 f"  Shape={probabilities.shape}, dtype={probabilities.dtype}\n"
                 f"  Min={np.min(probabilities):.4f}, Max={np.max(probabilities):.4f}, Mean={np.mean(probabilities):.4f}\n"
                 f"  Sample values (first 5 from flattened): {probabilities.flatten()[:5]}")
    raw_mask = probabilities # Continue processing with probabilities
    # --- End Sigmoid Application ---

    # Squeeze batch dimension if present (e.g. from (1,H,W,C) to (H,W,C) or (1,H,W) to (H,W))
    if raw_mask.ndim == 4 and raw_mask.shape[0] == 1: # (1,H,W,C) or (1,H,W,1)
        squeezed_mask = np.squeeze(raw_mask, axis=0) # Becomes (H,W,C) or (H,W,1)
        logging.info(f"PREDICT_SEG_DEBUG (postprocess_mask - squeezed_mask stats after np.squeeze(axis=0)):\n"
                     f"  Shape={squeezed_mask.shape}, dtype={squeezed_mask.dtype}\n"
                     f"  Min={np.min(squeezed_mask):.4f}, Max={np.max(squeezed_mask):.4f}, Mean={np.mean(squeezed_mask):.4f}")
    elif raw_mask.ndim == 3 and raw_mask.shape[0] == 1: # This case might be for (1, H, W) if model output is already squeezed for single class
        squeezed_mask = np.squeeze(raw_mask, axis=0) # Becomes (H,W)
        logging.info(f"PREDICT_SEG_DEBUG (postprocess_mask - squeezed_mask stats after np.squeeze(axis=0) from 3D input):\n"
                     f"  Shape={squeezed_mask.shape}, dtype={squeezed_mask.dtype}\n"
                     f"  Min={np.min(squeezed_mask):.4f}, Max={np.max(squeezed_mask):.4f}, Mean={np.mean(squeezed_mask):.4f}")
    else:
        squeezed_mask = raw_mask # Assumed to be (H,W,C) or (H,W) or (H,W,1) already
        logging.info(f"PREDICT_SEG_DEBUG (postprocess_mask - squeezed_mask (no squeeze applied)):\n"
                     f"  Shape={squeezed_mask.shape}, dtype={squeezed_mask.dtype}\n"
                     f"  Min={np.min(squeezed_mask):.4f}, Max={np.max(squeezed_mask):.4f}, Mean={np.mean(squeezed_mask):.4f}")

    model_output_h, model_output_w = squeezed_mask.shape[0], squeezed_mask.shape[1] # H, W of the model's direct output (after potential squeeze)

    # Process based on number of classes
    if num_classes == 1:
        # Binary classification (single channel output) - Apply threshold
        # If raw_mask is (H,W,1), take the channel. If (H,W) assume it's already the logits/probabilities for the positive class.
        current_mask_probs = squeezed_mask[..., 0].astype(np.float32, copy=True)

        logging.info(f"PREDICT_SEG_DEBUG (postprocess_mask - current_mask_probs stats):\n"
                     f"  Shape={current_mask_probs.shape}, dtype={current_mask_probs.dtype}\n"
                     f"  Min={np.min(current_mask_probs):.4f}, Max={np.max(current_mask_probs):.4f}, Mean={np.mean(current_mask_probs):.4f}")

        actual_threshold_to_use = threshold  # Use the function's threshold parameter
        logging.info(f"  DEBUG: Threshold value being used: {actual_threshold_to_use:.4f} (from function arg)")

        is_any_above = np.any(current_mask_probs > actual_threshold_to_use)
        logging.info(f"  DEBUG: Result of np.any(current_mask_probs > {actual_threshold_to_use:.4f}): {is_any_above}")

        count_above = np.sum(current_mask_probs > actual_threshold_to_use)
        logging.info(f"  DEBUG: Result of np.sum(current_mask_probs > {actual_threshold_to_use:.4f}): {count_above}")

        if is_any_above and count_above < 20 and count_above > 0: # If a few pixels, print them
            true_indices = np.argwhere(current_mask_probs > actual_threshold_to_use)
            sample_values_str = "["
            for i in range(min(len(true_indices), 5)):
                idx_pair = true_indices[i]
                sample_values_str += f"({idx_pair[0]},{idx_pair[1]}): {current_mask_probs[idx_pair[0], idx_pair[1]]:.4f} "
            sample_values_str += "]"
            logging.info(f"  DEBUG: Sample high values (coord: val) above {actual_threshold_to_use:.4f}: {sample_values_str}")

        processed_mask = (current_mask_probs > actual_threshold_to_use).astype(np.uint8)
    elif num_classes > 1 and raw_mask.ndim == 3:
        # Multi-class classification - argmax
        processed_mask = np.argmax(raw_mask, axis=-1).astype(np.uint8) # (h, w)
    else:
        logging.error(f"PREDICT_SEG_DEBUG (postprocess_mask - Unexpected combination for processing: num_classes={num_classes}, raw_mask.shape={raw_mask.shape}. Defaulting to basic thresholding on raw_mask or first channel if 3D.")
        # Fallback: attempt basic thresholding
        mask_to_threshold = raw_mask[..., 0] if raw_mask.ndim == 3 and raw_mask.shape[-1] > 0 else raw_mask
        processed_mask = (mask_to_threshold > threshold).astype(np.uint8)

    logging.info(f"PREDICT_SEG_DEBUG (postprocess_mask - processed_mask BEFORE resize):\n"
                 f"  Shape={processed_mask.shape}, dtype={processed_mask.dtype}\n"
                 f"  Min={processed_mask.min()}, Max={processed_mask.max()}, Non-zero pixels={np.count_nonzero(processed_mask)}")

    # Resize mask back to target size using NEAREST
    # OpenCV resize takes (w, h)
    target_size_wh = (target_size_hw[1], target_size_hw[0])
    final_mask = cv2.resize(processed_mask, target_size_wh, interpolation=cv2.INTER_NEAREST)
    logging.info(f"PREDICT_SEG_DEBUG (postprocess_mask - final_mask AFTER resize to {target_size_hw}):\n"
                 f"  Shape={final_mask.shape}, dtype={final_mask.dtype}\n"
                 f"  Min={final_mask.min()}, Max={final_mask.max()}, Non-zero pixels={np.count_nonzero(final_mask)}")
    return final_mask

def overlay_mask_on_image(image_path:str, mask:np.ndarray, alpha:float=0.5) -> np.ndarray|None:
    """Overlays the predicted mask on the original image."""
    try:
        original_image = cv2.imread(image_path)
        if original_image is None:
            logging.error(f"Failed to read original image for overlay: {image_path}")
            return None
        # Ensure mask and image has the same (h, w)
        if original_image.shape[:2] != mask.shape[:2]:
            logging.warning(f"Original image size {original_image.shape[:2]} and mask size {mask.shape[:2]} differ. Resizing mask for overlay")
            mask = cv2.resize(mask.astype(np.uint8), (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
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
        overlayed_image = cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB)
        return overlayed_image
    except Exception as e:
        logging.error(f"Error overlaying mask: {e}")
        return None

# Core logic

def load_segmentation_model(tflite_model_path:str, expected_input_size_hw: tuple) -> tuple[tf.lite.Interpreter, list, list]:
    """Loads the TFLite segmentation model interpreter."""
    try:
        logging.info(f"Loading TFLite segmentation model from: {tflite_model_path}")
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors() # Allocate tensors directly
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        logging.info(f"Model input_details[0]['shape'] AFTER alloc: {input_details[0]['shape']}")
        logging.info(f"Model input_details[0]['dtype'] AFTER alloc: {input_details[0]['dtype']}")
            
        return interpreter, input_details, output_details
    except ValueError as e:
        logging.error(f"Failed to load TFLite interpreter or allocate tensors for {tflite_model_path}: {e}")
        raise

def run_segmentation_inference(
    interpreter:tf.lite.Interpreter,
    input_details:list, # These are now details AFTER potential resize and alloc
    output_details:list,
    image_path:str,
    model_input_size_hw:tuple, # e.g. (256,256) from config
    output_resize_shape_hw:tuple, # e.g., original image shape or depth map shape
    num_classes:int = 2,
    threshold:float = 0.5): # Add threshold parameter with a default
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
        threshold: Confidence threshold for binary segmentation (if num_classes <= 2).

    Returns:
        Processed segmentation mask (resized to output_resize_shape_hw) or None on failure.
    """
    try:
        logging.debug(f"Running inference for image: {image_path}")
        logging.debug(f"Model Input Size (H,W): {model_input_size_hw}, Output Resize Shape (H,W): {output_resize_shape_hw}, Num Classes: {num_classes}")

        # Load and preprocess image
        try:
            preprocessed_image, original_size_hw = load_and_preprocess_image(image_path, model_input_size_hw)
            if preprocessed_image is None:
                logging.error(f"Preprocessing failed for image: {image_path}")
                return None
            logging.debug(f"Image preprocessed successfully. Shape: {preprocessed_image.shape}, Dtype: {preprocessed_image.dtype}")
            logging.info(f"PREDICT_SEG_DEBUG (Preprocessed Image for TFLite): "
                         f"Shape={preprocessed_image.shape}, dtype={preprocessed_image.dtype}, "
                         f"Min={np.min(preprocessed_image)}, Max={np.max(preprocessed_image)}, "
                         f"Mean={np.mean(preprocessed_image)}, "
                         f"Sample values (first 5 from flattened): {preprocessed_image.flatten()[:5]}")
        except Exception as preproc_e:
            logging.exception(f"Error during preprocessing for {image_path}: {preproc_e}")
            return None

        input_index = input_details[0]['index']
        logging.info(f"--- Pre-set_tensor --- Input shape: {preprocessed_image.shape}, Expected by model: {input_details[0]['shape']}")
        
        expected_dtype = input_details[0]['dtype']
        if preprocessed_image.dtype != expected_dtype:
            logging.warning(f"Dtype mismatch! Casting from {preprocessed_image.dtype} to {expected_dtype}.")
            preprocessed_image = preprocessed_image.astype(expected_dtype)

        interpreter.set_tensor(input_index, preprocessed_image)
        t_invoke_start = time.time()
        logging.debug("Invoking TFLite interpreter...")
        try:
            interpreter.invoke()
            logging.debug("Interpreter invoked successfully.")
        except Exception as invoke_e:
            logging.exception(f"Error invoking TFLite interpreter for {image_path}: {invoke_e}")
            return None

        # Get output tensor
        output_index = output_details[0]['index']
        logging.debug(f"Getting output tensor from index {output_index}.")
        try:
            raw_output = interpreter.get_tensor(output_index)
            logging.info(f"PREDICT_SEG_DEBUG (Raw Model Output - from interpreter.get_tensor()):\n"
                         f"  Shape={raw_output.shape}, dtype={raw_output.dtype}\n"
                         f"  Min={np.min(raw_output):.4f}, Max={np.max(raw_output):.4f}, Mean={np.mean(raw_output):.4f}\n"
                         f"  Sample values (first 5 from flattened): {raw_output.flatten()[:5]}")
            logging.debug(f"Raw output tensor shape: {raw_output.shape}, dtype: {raw_output.dtype}")
        except Exception as get_tensor_e:
            logging.exception(f"Error getting output tensor for {image_path}: {get_tensor_e}")
            return None

        # Postprocess mask
        final_mask = postprocess_mask(raw_output, output_resize_shape_hw, num_classes, threshold=threshold) # Pass the threshold
        return final_mask
    except Exception as e:
        logging.exception(f"Error during segmentation inference for {image_path}: {e}")
        return None

def predict_standalone(config_path:str, image_path:str, output_path:str = None, show_overlay:bool=False):
    """Loads TFLite model and performs segmentation prediction (Standalone mode)."""
    try:
        project_root = get_project_root()
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
        # Read input size from the 'model' section, taking Height and Width
        model_input_size_hw = tuple(config["model"]["input_shape"][:2]) 
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
        interpreter, input_details, output_details = load_segmentation_model(tflite_model_path, model_input_size_hw)
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
        logging.error("Segmentation inference failed")
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