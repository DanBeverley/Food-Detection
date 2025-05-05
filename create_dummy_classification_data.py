import os
import json
from PIL import Image
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
METADATA_PATH = PROJECT_ROOT / "data" / "dummy_classification" / "metadata.json"
PROCESSED_DIR = PROJECT_ROOT / "data" / "dummy_classification" / "processed"
CLASSES = ["dummy_class_a", "dummy_class_b"]
IMAGES_PER_CLASS = 4
IMAGE_SIZE = (32, 32) 

def create_dummy_data():
    print("Creating dummy classification data...")
    metadata = []

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    for class_name in CLASSES:
        class_dir = PROCESSED_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Creating directory: {class_dir}")

        for i in range(IMAGES_PER_CLASS):
            image_name = f"dummy_{class_name}_{i+1}.png"
            image_path = class_dir / image_name
            abs_image_path_str = str(image_path.resolve())

            # Create a simple black image
            img = Image.new('RGB', IMAGE_SIZE, color = 'black')
            img.save(image_path)
            print(f"    Created dummy image: {image_path}")

            # Add entry to metadata
            metadata.append({
                "image_path": abs_image_path_str, 
                "label": class_name
            })

    try:
        with open(METADATA_PATH, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Successfully created metadata file: {METADATA_PATH}")
    except IOError as e:
        print(f"Error writing metadata file {METADATA_PATH}: {e}")

if __name__ == "__main__":
    create_dummy_data()
