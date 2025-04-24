import os
import argparse
import logging
import tensorflow as tf
from tensorflow import keras
from typing import Dict
import json
from data import load_test_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def evaluate_model(model_path:str, test_dataset:tf.data.Dataset) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.

    Args:
        model_path: Path to the saved model file (e.g., .h5).
        test_dataset: Test tf.data.Dataset.

    Returns:
        Dictionary with evaluation metrics (e.g., accuracy, loss).

    Raises:
        FileNotFoundError: If model file is missing.
        RuntimeError: If evaluation fails.
    """
    try:
        model = keras.models.load_model(model_path)
        results = model.evaluate(test_dataset, return_dict = True)
        logger.info(f"Evaluation result: {results}")
        return results
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(f"Evaluation error: {e}")
        raise
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate classification model on test data.")
    parser.add_argument('--metadata_path', required=True, help="Path to JSON metadata from data loading.")
    parser.add_argument('--data_dir', required=True, help="Root directory of processed images (e.g., data/classification/processed/).")
    parser.add_argument('--model_path', required=True, help="Path to saved model file (e.g., trained_models/classification/model.h5).")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument('--image_size', nargs=2, type=int, default=[224, 224], help="Image dimensions (height width).")
    args = parser.parse_args()
    
    try:
        test_dataset = load_test_data(args.metadata_path, args.data_dir, tuple(args.image_size), args.batch_size)
        metrics = evaluate_model(args.model_path, test_dataset)
        # Save metrics to a file for logging
        with open(os.path.join(os.path.dirname(args.model_path), 'evaluation_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info("Evaluation complete. Metrics saved.")
    except Exception as e:
        logger.error(f"Script error: {e}")
        exit(1)