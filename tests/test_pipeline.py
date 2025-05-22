import unittest
import subprocess
import json
import os
import tempfile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MAIN_SCRIPT_PATH = os.path.join(PROJECT_ROOT, 'main.py')
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config_pipeline.yaml') # Default config
TEST_CASES = [
    {
        "name": "apple_1_with_precomputed_mask",
        "inputs": {
            "image": "E:/_MetaFood3D_new_RGBD_videos/RGBD_videos/Apple/apple_1/original/0.jpg",
            "depth": "E:/_MetaFood3D_new_RGBD_videos/RGBD_videos/Apple/apple_1/depth/0.jpg",
            "mask_path": "E:/_MetaFood3D_new_RGBD_videos/RGBD_videos/Apple/apple_1/masks/0.jpg",
            "mesh_file_path": "E:/_MetaFood3D_new_3D_Mesh/3D_Mesh/Apple/apple_1/apple_1.obj",
            "point_cloud_file": "E:/_MetaFood3D_new_Point_cloud/Point_cloud/4096/Apple/apple_1/apple_1_sampled_1.ply",
            "config": CONFIG_PATH
        },
        "expected_outputs": {
            "food_label": "Almond(bowl)", # Current known misclassification
            "segmentation_source": "precomputed_mask_file: 0.jpg",
            "volume_cm3_approx": 247.94, # From last successful run
            "volume_tolerance_percent": 5.0
        }
    },
 
]

class TestFoodAnalysisPipeline(unittest.TestCase):

    def run_pipeline_test(self, test_case_data):
        """Helper function to run a single pipeline test case."""
        args = ['python', MAIN_SCRIPT_PATH]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_output_file:
            output_json_path = tmp_output_file.name
        
        for key, value in test_case_data['inputs'].items():
            if value is not None: 
                arg_name = f"--{key.replace('_', '-')}" 
                if key == "mask_path" or key == "mesh_file_path" or key == "point_cloud_file" or key == "config" or key=="image" or key=="depth":
                     arg_name = f"--{key}"

                args.extend([arg_name, str(value)])
        args.extend(['--output', output_json_path])
        args.append('--verbose') # For more detailed logs during test failures

        try:
            print(f"Running test: {test_case_data['name']}\nCommand: {' '.join(args)}")
            completed_process = subprocess.run(args, capture_output=True, text=True, check=False, cwd=PROJECT_ROOT)
            
            
            if completed_process.stdout:
                print("STDOUT:\n", completed_process.stdout)
            if completed_process.stderr:
                print("STDERR:\n", completed_process.stderr)
            
            completed_process.check_returncode() 

            self.assertTrue(os.path.exists(output_json_path), f"Output JSON file not found: {output_json_path}")
            with open(output_json_path, 'r') as f:
                results = json.load(f)
            
            self.assertIsNone(results.get('error_message'), 
                              f"Pipeline reported an error: {results.get('error_message')}")

            expected = test_case_data['expected_outputs']
            if 'food_label' in expected:
                self.assertEqual(results.get('food_label'), expected['food_label'])
            if 'segmentation_source' in expected:
                self.assertEqual(results.get('segmentation_source'), expected['segmentation_source'])
            if 'volume_cm3_approx' in expected and 'volume_tolerance_percent' in expected:
                volume_actual = results.get('volume_cm3')
                self.assertIsNotNone(volume_actual, "Volume not found in results.")
                volume_expected = expected['volume_cm3_approx']
                tolerance = expected['volume_tolerance_percent'] / 100.0
                self.assertAlmostEqual(volume_actual, volume_expected, delta=volume_expected * tolerance,
                                     msg=f"Volume {volume_actual} not within {expected['volume_tolerance_percent']}% of {volume_expected}")

        finally:
            if os.path.exists(output_json_path):
                os.remove(output_json_path)

    def test_all_defined_cases(self):
        if not TEST_CASES:
            self.skipTest("No test cases defined.")
            
        for i, test_case_data in enumerate(TEST_CASES):
            with self.subTest(name=test_case_data['name'], i=i):
                self.run_pipeline_test(test_case_data)

if __name__ == '__main__':
    unittest.main()
