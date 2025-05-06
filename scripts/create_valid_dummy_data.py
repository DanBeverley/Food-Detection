import os
from PIL import Image
import numpy as np

base_data_path_seg = 'data/segmentation'
base_data_path_depth = 'data/depth'
image_dir = os.path.join(base_data_path_seg, 'images')
mask_dir = os.path.join(base_data_path_seg, 'masks')
depth_dir = base_data_path_depth
dummy_files = ['dummy_img_1', 'dummy_img_2']

os.makedirs(image_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)

img = Image.new('RGB', (1, 1), color = 'black')
mask_img = Image.new('L', (1, 1), color = 0)

dummy_depth_shape = (256, 256)
dummy_depth_value = 500
dummy_depth_map = np.full(dummy_depth_shape, dummy_depth_value, dtype=np.uint16)

for base_filename in dummy_files:
    img_path = os.path.join(image_dir, f"{base_filename}.png")
    mask_path = os.path.join(mask_dir, f"{base_filename}.png")
    depth_path = os.path.join(depth_dir, f"{base_filename}.npy")

    img.save(img_path, 'PNG')
    mask_img.save(mask_path, 'PNG')
    np.save(depth_path, dummy_depth_map)

    print(f"Created dummy image file: {img_path}")
    print(f"Created dummy mask file: {mask_path}")
    print(f"Created dummy depth file: {depth_path}")

print("\nDummy data creation complete.")
