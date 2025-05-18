import argparse
import os
import logging
import json 
import sys
from pathlib import Path

project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

try:
    from food_analyzer import analyze_food_item, load_pipeline_config
except ImportError as e:
    print(f"Error: Failed to import food_analyzer: {e}", file=sys.stderr)
    print("Ensure food_analyzer.py exists and project structure is correct.", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Food Analysis Pipeline - Main Entry Point")
    parser.add_argument("--image", required=True, help="Path to the input RGB image file.")
    parser.add_argument("--depth", required=False, help="Path to the input depth map file (.npy or image format).")
    parser.add_argument("--mesh_file_path", help="Optional path to a 3D mesh file for volume calculation.")
    parser.add_argument("--point_cloud_file", help="Optional path to a 3D point cloud file (e.g., .ply) for volume calculation.")
    parser.add_argument("--known_food_class", help="Optional known food class string (e.g., 'Apple').")
    parser.add_argument("--config", default="config_pipeline.yaml",
                        help="Path to the pipeline configuration YAML file (relative to project root).")
    parser.add_argument("--output", help="Optional path to save the results as a JSON file.")
    parser.add_argument("--mask_path", help="Optional path to a pre-computed segmentation mask image.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Debug logging enabled.")

    # Validate paths early
    config_abs_path = None
    if args.config:
        config_abs_path = Path(args.config).resolve()
        if not config_abs_path.is_file():
            logging.error(f"Error: Pipeline configuration file '{args.config}' not found at resolved path '{config_abs_path}'.")
            return # Exit if config is essential and not found

    try:
        # Load configuration from the validated absolute path
        config = load_pipeline_config(str(config_abs_path)) if config_abs_path else {}

        # Resolve other paths if provided
        image_abs_path = Path(args.image).resolve() if args.image else None
        depth_abs_path = Path(args.depth).resolve() if args.depth else None
        mesh_abs_path = Path(args.mesh_file_path).resolve() if args.mesh_file_path else None
        point_cloud_abs_path = Path(args.point_cloud_file).resolve() if args.point_cloud_file else None
        mask_abs_path = Path(args.mask_path).resolve() if args.mask_path else None

        # Validate existence of input files if paths are provided
        if image_abs_path and not image_abs_path.is_file():
            logging.error(f"Error: Image file not found at '{image_abs_path}'.")
            return
        if depth_abs_path and not depth_abs_path.is_file():
            logging.error(f"Error: Depth map file not found at '{depth_abs_path}'.")
            return
        if mesh_abs_path and not mesh_abs_path.is_file():
            logging.error(f"Error: Mesh file not found at '{mesh_abs_path}'.")
            return
        if point_cloud_abs_path and not point_cloud_abs_path.is_file():
            logging.error(f"Error: Point cloud file not found at '{point_cloud_abs_path}'.")
            return
        if mask_abs_path and not mask_abs_path.is_file():
            logging.error(f"Error: Mask file not found at '{mask_abs_path}'.")
            return

        # Load USDA API key from environment variable
        usda_api_key = os.getenv('USDA_API_KEY')
        if not usda_api_key and config.get('enable_nutrition_api', False):
            logging.warning("USDA_API_KEY environment variable not set. Nutritional analysis will be skipped.")

        logging.info("Starting full food analysis pipeline...")
        analysis_results = analyze_food_item(
            image_path=str(image_abs_path) if image_abs_path else None, # Pass string path or None
            config_path=str(config_abs_path) if config_abs_path else None, # Use validated absolute path or None
            depth_map_path=str(depth_abs_path) if depth_abs_path else None, # Use validated absolute path or None
            mesh_file_path=str(mesh_abs_path) if mesh_abs_path else None, # Use validated absolute path or None
            point_cloud_file_path=str(point_cloud_abs_path) if point_cloud_abs_path else None, # Use validated absolute path or None
            known_food_class=args.known_food_class,
            usda_api_key=usda_api_key, # Pass the key read from env
            mask_path=str(mask_abs_path) if mask_abs_path else None # Use validated absolute path or None
        )

        if analysis_results:
            print("\n--- Analysis Results ---")
            print(json.dumps(analysis_results, indent=2))

            if args.output:
                output_path = Path(args.output).resolve()
                output_dir = output_path.parent
                try:
                    if output_dir:
                        output_dir.mkdir(parents=True, exist_ok=True)
                    with open(output_path, 'w') as f:
                        json.dump(analysis_results, f, indent=2)
                    logging.info(f"Results saved to: {output_path}")
                except Exception as e_save:
                    logging.error(f"Failed to save results to {output_path}: {e_save}")
        else:
            logging.error("Analysis failed. No results to display or save.")

    except FileNotFoundError as e: # This will now catch if config_abs_path was None and load_pipeline_config tried to use it, or other FNFE
        logging.error(f"File Not Found Error during pipeline execution: {e}")
        # Config loading error already logged in helper or path validation
        # print(f"Error: Pipeline configuration file '{args.config}' not found.", file=sys.stderr)
    except ImportError as e:
        logging.error(f"Import Error: {e}. Check module paths and refactoring.")
        # print(f"Import Error: {e}. Check module paths and refactoring.", file=sys.stderr)
    except NotImplementedError as e:
        logging.error(f"Not Implemented Error: {e}. Prediction scripts likely need refactoring.")
        # print(f"Not Implemented Error: {e}. Prediction scripts likely need refactoring.", file=sys.stderr)
    except Exception as e:
        logging.exception(f"An unexpected error occurred in the main execution: {e}") # Use exception for full traceback


if __name__ == "__main__":
    main()