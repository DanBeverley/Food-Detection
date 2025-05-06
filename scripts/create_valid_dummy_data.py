import os
from PIL import Image

base_data_path = 'data/segmentation'
image_dir = os.path.join(base_data_path, 'images')
mask_dir = os.path.join(base_data_path, 'masks')
dummy_files = ['dummy_img_1.png', 'dummy_img_2.png']

os.makedirs(image_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

# Create a 1x1 black pixel image
img = Image.new('RGB', (1, 1), color = 'black')
# Create a 1x1 black pixel grayscale image for mask
mask_img = Image.new('L', (1, 1), color = 0) # 'L' mode is grayscale

# Save dummy images
for filename in dummy_files:
    img_path = os.path.join(image_dir, filename)
    mask_path = os.path.join(mask_dir, filename)

    img.save(img_path, 'PNG')
    mask_img.save(mask_path, 'PNG')
    print(f"Created valid dummy file: {img_path}")
    print(f"Created valid dummy file: {mask_path}")

print("Dummy data creation complete.")
