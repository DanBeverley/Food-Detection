import os
import argparse
import logging
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, Dict
from data import load_segmentation_test_data  

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_segmentation_model(model_path: str, test_dataset: tf.data.Dataset) -> Dict[str, float]:
    """
    Evaluate a trained segmentation model on test data, using metrics like accuracy and IoU.

    Args:
        model_path: Path to the saved model file (e.g., .h5).
        test_dataset: Test tf.data.Dataset yielding (image, mask) pairs.

    Returns:
        Dictionary with evaluation metrics.

    Raises:
        FileNotFoundError: If model file is missing.
        RuntimeError: If evaluation fails.
    """
    try:
        model = keras.models.load_model(model_path)
        
        # Custom metric for IoU (Intersection over Union) if not compiled in the model
        def mean_iou(y_true, y_pred):
            y_pred = tf.argmax(y_pred, axis=-1)
            y_true = tf.cast(y_true, tf.int32)
            intersection = tf.reduce_sum(y_true * tf.cast(tf.equal(y_true, y_pred), y_true.dtype))
            union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
            return tf.reduce_mean(intersection / (union + tf.keras.backend.epsilon()))
        
        results = model.evaluate(test_dataset, return_dict=True)
        logger.info(f"Evaluation results: {results}")
        return results
    
    except (FileNotFoundError, tf.errors.NotFoundError) as e:
        logger.error(f"Model loading error: {e}")
        raise
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate segmentation model on test data.")
    parser.add_argument('--metadata_path', required=True, help="Path to JSON metadata from data loading.")
    parser.add_argument('--data_dir', required=True, help="Root directory of processed images and masks (e.g., data/segmentation/processed/).")
    parser.add_argument('--model_path', required=True, help="Path to saved model file (e.g., trained_models/segmentation/model.h5).")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for evaluation.")
    parser.add_argument('--image_size', nargs=2, type=int, default=[512, 512], help="Image and mask dimensions (height width).")
    args = parser.parse_args()
    
    try:
        test_ds = load_segmentation_test_data(args.metadata_path, args.data_dir, tuple(args.image_size), args.batch_size)
        metrics = evaluate_segmentation_model(args.model_path, test_ds)
        # Save metrics to a file
        with open(os.path.join(os.path.dirname(args.model_path), 'evaluation_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info("Evaluation complete. Metrics saved.")
    except Exception as e:
        logger.error(f"Script error: {e}")
        exit(1)