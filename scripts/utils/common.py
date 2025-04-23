import os
import json
from PIL import Image

def resize_image(input_path, output_path, size=(224,224)):
    """Resize image and save to output_path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img = Image.open(input_path).convert('RGB')
    img = img.resize(size)
    img.save(output_path)


def save_json(data, output_json):
    """Save Python object as JSON."""
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(json_path):
    """Load JSON file and return data."""
    with open(json_path) as f:
        return json.load(f)
