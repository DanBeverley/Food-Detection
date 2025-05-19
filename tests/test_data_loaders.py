import yaml
import tensorflow as tf
import sys
import os
import pathlib
import traceback

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from models.classification.data import load_classification_data
    from models.segmentation.data import load_segmentation_data
except ImportError as e:
    print(f"Error importing data loaders: {e}")
    print("Please ensure that the script is run from the project root or that PYTHONPATH is set correctly.")
    sys.exit(1)

def load_config(config_path: pathlib.Path):
    """Loads a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A dictionary with the loaded configuration or None if an error occurs.
    """
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        return None
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def print_batch_structure(batch_name, inputs, labels_or_mask):
    print(f"\n--- {batch_name} Batch Structure ---")
    if isinstance(inputs, dict):
        print("Inputs (Dictionary):")
        for key, tensor in inputs.items():
            print(f"  Input '{key}': shape={tensor.shape}, dtype={tensor.dtype.name}")
    elif hasattr(inputs, 'shape') and hasattr(inputs, 'dtype'):
        print(f"Inputs: shape={inputs.shape}, dtype={inputs.dtype.name}")
    else:
        print(f"Inputs: type={type(inputs)}, value={inputs}")

    if isinstance(labels_or_mask, dict):
        print("Labels/Mask (Dictionary):")
        for key, tensor in labels_or_mask.items():
            print(f"  Label/Mask '{key}': shape={tensor.shape}, dtype={tensor.dtype.name}")
    elif hasattr(labels_or_mask, 'shape') and hasattr(labels_or_mask, 'dtype'):
         print(f"Labels/Mask: shape={labels_or_mask.shape}, dtype={labels_or_mask.dtype.name}")
    else:
        print(f"Labels/Mask: type={type(labels_or_mask)}, value={labels_or_mask}")
    print("-" * 40)

def main():
    print(f"Using TensorFlow version: {tf.__version__}")
    print(f"Project Root determined as: {PROJECT_ROOT}")

    classification_config_path = PROJECT_ROOT / "models" / "classification" / "config.yaml"
    segmentation_config_path = PROJECT_ROOT / "models" / "segmentation" / "config.yaml"

    print(f"\nAttempting to load Classification Config from: {classification_config_path}")
    cls_config = load_config(classification_config_path)
    if cls_config:
        try:
            print("\nTesting Classification Data Loader...")
            print("Calling load_classification_data...")
            train_cls_dataset, val_cls_dataset, test_cls_dataset, cls_class_names, num_cls_classes = load_classification_data(cls_config)

            if train_cls_dataset:
                print("Classification training dataset object created.")
                print("Attempting to take 1 batch from classification training dataset...")
                try:
                    for inputs, labels in train_cls_dataset.take(1):
                        print_batch_structure("Classification Training", inputs, labels)
                    print("Successfully took 1 batch from classification training dataset.")
                except Exception as e_batch:
                    print(f"Error taking batch from classification training dataset: {e_batch}")
                    traceback.print_exc()
                    
                print(f"Number of classification classes: {num_cls_classes}")
                print(f"Classification class names: {cls_class_names}")
            else:
                print("Failed to load classification training dataset.")

        except Exception as e:
            print(f"Error testing classification data loader: {e}")
            traceback.print_exc()
    else:
        print("Skipping classification data loader test due to missing config.")

    # Test Segmentation Data Loader
    print(f"\nAttempting to load Segmentation Config from: {segmentation_config_path}")
    seg_config = load_config(segmentation_config_path)
    if seg_config:
        try:
            print("\nTesting Segmentation Data Loader...")
            train_seg_dataset, val_seg_dataset, test_seg_dataset, num_seg_classes = load_segmentation_data(seg_config)
            
            if train_seg_dataset:
                print("Segmentation training dataset loaded successfully.")
                for inputs, masks in train_seg_dataset.take(1):
                    print_batch_structure("Segmentation Training", inputs, masks)
                print(f"Number of segmentation classes: {num_seg_classes}")
            else:
                print("Failed to load segmentation training dataset.")
                
        except Exception as e:
            print(f"Error testing segmentation data loader: {e}")
            traceback.print_exc()
    else:
        print("Skipping segmentation data loader test due to missing config.")

if __name__ == "__main__":
    main()
