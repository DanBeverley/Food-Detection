#!/usr/bin/env python3
"""
Production Training Pipeline for Food Detection System

Orchestrates the complete training and export pipeline for both classification and segmentation models.
Designed for production environments including cloud platforms like Kaggle.

Author: Food Detection System
Version: 2.0
"""

import subprocess
import os
import sys
import logging
from pathlib import Path

# Configure logging for production
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Ensure project root is in Python path for imports
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configuration paths
CLASSIFICATION_CONFIG_PATH = "models/classification/config.yaml"
SEGMENTATION_CONFIG_PATH = "models/segmentation/config.yaml"

def run_training_step(model_name: str, script_path: str, config_path: str) -> bool:
    """
    Execute a training or export script as a subprocess with robust error handling.
    
    Args:
        model_name: Human-readable name for the training step
        script_path: Path to the Python script to execute
        config_path: Path to the configuration file
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"=== Starting {model_name} ===")
    
    # Construct command
    command = [sys.executable, script_path, "--config", config_path]
    logger.info(f"Executing: {' '.join(command)}")
    
    try:
        # Set environment for optimal subprocess execution
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"  # Ensure real-time output
        
        # Execute subprocess with real-time output capture
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
            cwd=project_root
        )
        
        # Stream output in real-time
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                if line.strip():  # Only log non-empty lines
                    print(f"[{model_name}] {line.strip()}", flush=True)
        
        # Wait for process completion
        process.wait()
        
        # Clean up
        if process.stdout:
            process.stdout.close()
        
        # Check execution result
        if process.returncode == 0:
            logger.info(f"=== {model_name} completed successfully ===")
            return True
        else:
            logger.error(f"=== {model_name} failed with exit code {process.returncode} ===")
            return False
            
    except FileNotFoundError:
        logger.error(f"Script not found: '{script_path}'. Verify the path is correct.")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during {model_name}: {e}")
        return False

def verify_prerequisites() -> bool:
    """
    Verify that all required files and directories exist before starting training.
    
    Returns:
        bool: True if all prerequisites are met
    """
    logger.info("=== Verifying Prerequisites ===")
    
    required_files = [
        CLASSIFICATION_CONFIG_PATH,
        SEGMENTATION_CONFIG_PATH,
        "models/classification/train.py",
        "models/classification/export_tflite.py", 
        "models/segmentation/train.py",
        "models/segmentation/export_tflite.py"
    ]
    
    required_dirs = [
        "data/classification",
        "data/segmentation",
        "trained_models"
    ]
    
    # Check required files
    missing_files = []
    for file_path in required_files:
        if not (project_root / file_path).exists():
            missing_files.append(file_path)
    
    # Check required directories
    missing_dirs = []
    for dir_path in required_dirs:
        if not (project_root / dir_path).exists():
            missing_dirs.append(dir_path)
    
    # Report results
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
    
    if missing_dirs:
        logger.error(f"Missing required directories: {missing_dirs}")
    
    success = len(missing_files) == 0 and len(missing_dirs) == 0
    
    if success:
        logger.info("All prerequisites verified successfully")
    else:
        logger.error("Prerequisites check failed. Please ensure all required files exist.")
    
    return success

def main():
    """
    Main training pipeline orchestrator.
    Executes the complete training and export sequence for production deployment.
    """
    logger.info("="*60)
    logger.info("PRODUCTION TRAINING PIPELINE - FOOD DETECTION SYSTEM")
    logger.info("="*60)
    
    # Verify prerequisites
    if not verify_prerequisites():
        logger.error("Prerequisites check failed. Aborting pipeline.")
        sys.exit(1)
    
    # Track pipeline success
    pipeline_steps = []
    
    # Step 1: Classification Model Training
    logger.info("\n" + "="*50)
    logger.info("STEP 1: CLASSIFICATION MODEL TRAINING")
    logger.info("="*50)
    
    class_train_success = run_training_step(
        "Classification Training",
        "models/classification/train.py",
        CLASSIFICATION_CONFIG_PATH
    )
    pipeline_steps.append(("Classification Training", class_train_success))
    
    if not class_train_success:
        logger.error("Classification training failed. Pipeline cannot continue.")
        sys.exit(1)
    
    # Step 2: Classification Model Export
    logger.info("\n" + "="*50)
    logger.info("STEP 2: CLASSIFICATION MODEL TFLITE EXPORT")
    logger.info("="*50)
    
    class_export_success = run_training_step(
        "Classification TFLite Export",
        "models/classification/export_tflite.py",
        CLASSIFICATION_CONFIG_PATH
    )
    pipeline_steps.append(("Classification Export", class_export_success))
    
    if not class_export_success:
        logger.warning("Classification TFLite export failed. Continuing with segmentation training.")
    
    # Step 3: Segmentation Model Training
    logger.info("\n" + "="*50)
    logger.info("STEP 3: SEGMENTATION MODEL TRAINING")
    logger.info("="*50)
    
    seg_train_success = run_training_step(
        "Segmentation Training",
        "models/segmentation/train.py",
        SEGMENTATION_CONFIG_PATH
    )
    pipeline_steps.append(("Segmentation Training", seg_train_success))
    
    if not seg_train_success:
        logger.error("Segmentation training failed. Pipeline cannot continue.")
        sys.exit(1)
    
    # Step 4: Segmentation Model Export
    logger.info("\n" + "="*50)
    logger.info("STEP 4: SEGMENTATION MODEL TFLITE EXPORT")
    logger.info("="*50)
    
    seg_export_success = run_training_step(
        "Segmentation TFLite Export",
        "models/segmentation/export_tflite.py",
        SEGMENTATION_CONFIG_PATH
    )
    pipeline_steps.append(("Segmentation Export", seg_export_success))
    
    if not seg_export_success:
        logger.warning("Segmentation TFLite export failed. Production deployment may be affected.")
    
    # Pipeline Summary
    logger.info("\n" + "="*60)
    logger.info("PRODUCTION TRAINING PIPELINE SUMMARY")
    logger.info("="*60)
    
    all_successful = True
    for step_name, success in pipeline_steps:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"{step_name:.<40} {status}")
        if not success:
            all_successful = False
    
    logger.info("="*60)
    
    if all_successful:
        logger.info("🎉 PRODUCTION PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("All models trained and exported. System ready for production deployment.")
    else:
        logger.warning("⚠️ PIPELINE COMPLETED WITH WARNINGS")
        logger.warning("Some steps failed. Review logs and address issues before production deployment.")
    
    logger.info("="*60)
    
    # Provide next steps
    logger.info("\nNext Steps:")
    logger.info("1. Verify TFLite models in trained_models/*/exported/")
    logger.info("2. Test inference pipeline with sample data")
    logger.info("3. Deploy to production environment")
    
    return 0 if all_successful else 1

if __name__ == "__main__":
    sys.exit(main())