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
    parser.add_argument("--known_food_class", help="Optional known food class string (e.g., 'Apple').")
    parser.add_argument("--config", default="config_pipeline.yaml",
                        help="Path to the pipeline configuration YAML file (relative to project root).")
    parser.add_argument("--output", help="Optional path to save the results as a JSON file.")
    parser.add_argument("--api_key", default=os.environ.get("USDA_API_KEY"),
                        help="USDA API Key for density lookup. Defaults to USDA_API_KEY environment variable.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Debug logging enabled.")

    if not os.path.exists(args.image):
        logging.error(f"Input image not found: {args.image}")
        return

    try:
        # Load pipeline configuration
        config_abs_path = os.path.join(project_root, args.config)
        config = load_pipeline_config(config_abs_path)

        # Run the analysis
        analysis_results = analyze_food_item(
            image_path=args.image,
            config=config,
            depth_map_path=args.depth,
            mesh_file_path=args.mesh_file_path,
            known_food_class=args.known_food_class,
            usda_api_key=args.api_key
        )

        if analysis_results:
            # Remove mask shape for cleaner printing/saving if present
            results_to_print = analysis_results.copy()
            # results_to_print.pop('segmentation_mask_shape', None) # Keep shape for info

            print("\n--- Analysis Results ---")
            # Use json for consistent formatting, handling None values correctly
            print(json.dumps(results_to_print, indent=2))

            # Save results if output path provided
            if args.output:
                try:
                    # Ensure output directory exists
                    output_dir = os.path.dirname(args.output)
                    if output_dir:
                         os.makedirs(output_dir, exist_ok=True)
                    # Save results dictionary as JSON
                    with open(args.output, 'w') as f:
                        json.dump(results_to_print, f, indent=2)
                    logging.info(f"Results saved to: {args.output}")
                except Exception as e:
                    logging.error(f"Failed to save results to {args.output}: {e}")
        else:
            print("\n--- Analysis Failed ---")

    except FileNotFoundError:
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