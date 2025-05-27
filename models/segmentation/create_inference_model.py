import os
import sys
import tensorflow as tf
from tensorflow import keras
import logging

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import dice_loss, focal_loss, combined_loss, BinaryIoU, DiceCoefficient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_inference_model():
    """Create an inference-only model without custom loss functions."""
    
    # Path to the trained model
    model_path = "../../trained_models/segmentation/unet_segmentation_final_20250527-151342.h5"
    
    # Custom objects for loading
    custom_objects = {
        'dice_loss': dice_loss,
        'focal_loss': focal_loss,
        'combined_loss': combined_loss,
        'BinaryIoU': BinaryIoU,
        'DiceCoefficient': DiceCoefficient,
    }
    
    logger.info(f"Loading model from: {model_path}")
    
    # Load the model with custom objects
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    
    logger.info("Model loaded successfully")
    
    # Create a new model with the same architecture but without custom loss/metrics
    # This will only keep the weights and architecture
    inference_model = keras.Model(inputs=model.input, outputs=model.output)
    
    # Save the inference model without custom objects
    inference_path = "../../trained_models/segmentation/unet_inference_model.h5"
    inference_model.save(inference_path)
    
    logger.info(f"Inference model saved to: {inference_path}")
    
    return inference_path

if __name__ == "__main__":
    create_inference_model() 