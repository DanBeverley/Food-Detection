from ultralytics import YOLO
import argparse
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_yolo(data_yaml_path, epochs=100, imgsz=640, model_size='n', project_dir='runs/segment'):
    """
    Trains a YOLOv8-seg model.
    
    Args:
        data_yaml_path: Path to dataset.yaml
        epochs: Number of training epochs
        imgsz: Image size
        model_size: Model size ('n', 's', 'm', 'l', 'x')
        project_dir: Directory to save runs
    """
    model_name = f"yolov8{model_size}-seg.pt"
    logging.info(f"Loading model: {model_name}")
    
    # Load model
    model = YOLO(model_name)
    
    logging.info(f"Starting training for {epochs} epochs...")
    
    # Train
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        patience=20,
        batch=16,
        project=project_dir,
        name=f"food_seg_{model_size}",
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        lr0=1e-3,
        seed=42
    )
    
    logging.info("Training complete.")
    
    # Export to TFLite for mobile
    logging.info("Exporting to TFLite...")
    success = model.export(format="tflite", int8=False) # int8=True for quantization, requires calibration data
    
    if success:
        logging.info(f"Export successful: {success}")
    else:
        logging.error("Export failed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8-seg on Food Dataset")
    parser.add_argument("--data", required=True, help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--size", type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], help="Model size")
    parser.add_argument("--project", type=str, default="runs/segment", help="Project output directory")
    
    args = parser.parse_args()
    
    train_yolo(args.data, args.epochs, args.imgsz, args.size, args.project)
