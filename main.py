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

    # Get API key from environment variable
    usda_api_key = os.environ.get('USDA_API_KEY')
    if not usda_api_key:
        logging.info("USDA_API_KEY environment variable not set. USDA API lookups will be skipped if data not in custom DB.")

    # --- Input File Validation ---
    # Required: config file
    config_abs_path = os.path.abspath(args.config)
    if not os.path.exists(config_abs_path):
        logging.error(f"Configuration file not found: {config_abs_path}")
        sys.exit(1)

    # Required: image file
    image_abs_path = None
    if args.image:
        image_abs_path = os.path.abspath(args.image)
        if not os.path.exists(image_abs_path):
            logging.error(f"Input image file not found: {image_abs_path}")
            sys.exit(1)
    else:
        logging.error("No input image provided. Please specify --image.")
        sys.exit(1)

    # Optional files: depth, mesh, point cloud, mask
    depth_abs_path = None
    if args.depth:
        depth_abs_path = os.path.abspath(args.depth)
        if not os.path.exists(depth_abs_path):
            logging.warning(f"Depth map file not found: {depth_abs_path}. Proceeding without it.")
            depth_abs_path = None

    mesh_abs_path = None
    if args.mesh_file_path:
        mesh_abs_path = os.path.abspath(args.mesh_file_path)
        if not os.path.exists(mesh_abs_path):
            logging.warning(f"Mesh file not found: {mesh_abs_path}. Proceeding without it.")
            mesh_abs_path = None

    point_cloud_abs_path = None
    if args.point_cloud_file:
        point_cloud_abs_path = os.path.abspath(args.point_cloud_file)
        if not os.path.exists(point_cloud_abs_path):
            logging.warning(f"Point cloud file not found: {point_cloud_abs_path}. Proceeding without it.")
            point_cloud_abs_path = None

    mask_abs_path = None
    if args.mask_path:
        mask_abs_path = os.path.abspath(args.mask_path)
        if not os.path.exists(mask_abs_path):
            logging.warning(f"Pre-computed mask file not found: {mask_abs_path}. Mask will be model-generated if needed.")
            mask_abs_path = None
    # --- End Input File Validation ---

    # Load configuration (already validated path)
    config = load_pipeline_config(config_abs_path)

    # Run the analysis
    analysis_results = analyze_food_item(
        image_path=image_abs_path, # Use validated absolute path
        config_path=config_abs_path, # Use validated absolute path
        depth_map_path=depth_abs_path, # Use validated absolute path or None
        mesh_file_path=mesh_abs_path, # Use validated absolute path or None
        point_cloud_file_path=point_cloud_abs_path, # Use validated absolute path or None
        known_food_class=args.known_food_class,
        usda_api_key=usda_api_key, # Pass the key read from env
        mask_path=mask_abs_path # Use validated absolute path or None
    )

    if analysis_results:
        # Remove mask shape for cleaner printing/saving if present
        # results_to_print = analysis_results.copy() # Redundant copy
        # results_to_print.pop('segmentation_mask_shape', None) # Keep shape for info

        print("\n--- Analysis Results ---")
        # Use json for consistent formatting, handling None values correctly
        print(json.dumps(analysis_results, indent=2))

        # Save results if output path provided
        if args.output:
            try:
                # Ensure output directory exists
                output_dir = os.path.dirname(args.output)
                if output_dir:
                     os.makedirs(output_dir, exist_ok=True)
                # Save results dictionary as JSON
                with open(args.output, 'w') as f:
                    json.dump(analysis_results, f, indent=2)
                logging.info(f"Results saved to: {args.output}")
            except Exception as e:
                logging.error(f"Failed to save results to {args.output}: {e}")
    else:
        logging.error("Analysis failed. No results to display or save.")

except FileNotFoundError as e:
     # Config loading error already logged in helper
     print(f"Error: Pipeline configuration file '{args.config}' not found.", file=sys.stderr)
except ImportError as e:
     # Error importing refactored functions etc.
     print(f"Import Error: {e}. Check module paths and refactoring.", file=sys.stderr)
except NotImplementedError as e:
     # Error if refactoring wasn't done
     print(f"Not Implemented Error: {e}. Prediction scripts likely need refactoring.", file=sys.stderr)
except Exception as e:
    logging.exception(f"An unexpected error occurred in the main execution: {e}") # Use exception for full traceback


if __name__ == "__main__":
    main()