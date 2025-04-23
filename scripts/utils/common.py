from typing import Tuple, Any, Dict
from PIL import Image
import os
import json
from functools import lru_cache

def resize_image(input_path: str, output_path: str, size: Tuple[int, int] = (224, 224)) -> None:
    """
    Resize an image and save it, with caching for repeated calls.

    Args:
        input_path: Path to the input image file.
        output_path: Path to save the resized image.
        size: Tuple of (width, height) for resizing.

    Raises:
        FileNotFoundError: If the input file is missing.
        IOError: If saving the file fails.
    """
    try:
        @lru_cache(maxsize=50)
        def cached_resize(path: str, sz: Tuple[int, int]) -> Image.Image:
            img = Image.open(path).convert('RGB')
            return img.resize(sz)
        
        img = cached_resize(input_path, size)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)
    except (FileNotFoundError, IOError) as e:
        print(f"Error: {e}")
        raise

def save_json(data: Any, output_json: str) -> None:
    """
    Save data to a JSON file with error handling.

    Args:
        data: Python object to save (e.g., lists, dicts).
        output_json: Path to the output JSON file.

    Raises:
        IOError: If writing the file fails.
    """
    try:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(data, f, indent=2)
    except IOError as e:
        print(f"Error: {e}")
        raise

def load_json(json_path: str) -> Dict[str, Any]:
    """
    Load and return data from a JSON file.

    Args:
        json_path: Path to the JSON file.

    Returns:
        Loaded data as a dictionary.

    Raises:
        FileNotFoundError: If the file is missing.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: {e}")
        raise