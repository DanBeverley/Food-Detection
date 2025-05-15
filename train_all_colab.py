import subprocess
import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

CLASSIFICATION_CONFIG_PATH = "models/classification/config.yaml"
SEGMENTATION_CONFIG_PATH = "models/segmentation/config.yaml"

def run_training_step(model_name: str, script_path: str, config_path: str):
    """Runs a training or export script as a subprocess."""
    logging.info(f"--- Starting {model_name} --- ")    
    command = [sys.executable, script_path, "--config", config_path]
    
    logging.info(f"Executing command: {' '.join(command)}")
    try:

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                logging.info(f"[{model_name}] {line.strip()}")
        
        process.wait() 
        process.stdout.close() 

        if process.returncode == 0:
            logging.info(f"--- {model_name} completed successfully. ---")
            return True
        else:
            logging.error(f"--- {model_name} failed with exit code {process.returncode}. ---")
            return False
    except FileNotFoundError:
        logging.error(f"Error: The script at '{script_path}' was not found. Ensure the path is correct.")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred while running {model_name}: {e}")
        return False

def main():
    logging.info("===== Starting Full Training and Export Pipeline =====")
    
    logging.info("STEP 1: Training Classification Model")
    class_train_success = run_training_step(
        "Classification Training",
        "models/classification/train.py",
        CLASSIFICATION_CONFIG_PATH
    )
    if not class_train_success:
        logging.error("Classification training failed. Aborting further steps.")
        return

    logging.info("STEP 2: Exporting Classification Model to TFLite")
    class_export_success = run_training_step(
        "Classification TFLite Export",
        "models/classification/export_tflite.py",
        CLASSIFICATION_CONFIG_PATH
    )
    if not class_export_success:
        logging.warning("Classification TFLite export failed. Continuing with segmentation if possible, but pipeline will be incomplete.")

    logging.info("STEP 3: Training Segmentation Model")
    seg_train_success = run_training_step(
        "Segmentation Training",
        "models/segmentation/train.py",
        SEGMENTATION_CONFIG_PATH
    )
    if not seg_train_success:
        logging.error("Segmentation training failed. Aborting further steps.")
        return

    logging.info("STEP 4: Exporting Segmentation Model to TFLite")
    seg_export_success = run_training_step(
        "Segmentation TFLite Export",
        "models/segmentation/export_tflite.py",
        SEGMENTATION_CONFIG_PATH
    )
    if not seg_export_success:
        logging.warning("Segmentation TFLite export failed. The pipeline might not have a working segmentation model.")

    logging.info("===== Full Training and Export Pipeline Finished. =====")
    if class_train_success and class_export_success and seg_train_success and seg_export_success:
        logging.info("All steps completed successfully!")
    else:
        logging.warning("One or more steps in the pipeline failed. Please review logs.")

if __name__ == "__main__":
    main()