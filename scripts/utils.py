import os
import pathlib

def _get_project_root() -> pathlib.Path:
    """Returns the absolute path to the project root directory."""
    # Assuming this script (utils.py) is in Food-Detection/scripts/
    # So, project_root is the parent of the parent directory of this file.
    return pathlib.Path(__file__).resolve().parent.parent

if __name__ == '__main__':
    # Example of how to use it
    project_root = _get_project_root()
    print(f"Project Root: {project_root}")
    # Example: Constructing a path to a config file
    config_path = project_root / "config_pipeline.yaml"
    print(f"Example config path: {config_path}, Exists: {config_path.exists()}")
