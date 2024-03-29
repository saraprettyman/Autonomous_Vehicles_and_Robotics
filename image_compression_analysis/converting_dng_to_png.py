import rawpy
import imageio
import os
from skimage.transform import resize
import numpy as np

# Define the directories
dng_dir = 'photos/dng_originals/'
png_dir = 'photos/png_conversion/'

# Create the png directory if it doesn't exist
os.makedirs(png_dir, exist_ok=True)

# Loop through all the .dng files in the dng directory
for filename in os.listdir(dng_dir):
    if filename.endswith('.dng'):
        # Read the .dng file
        with rawpy.imread(os.path.join(dng_dir, filename)) as raw:
            # Convert it to an RGB image
            rgb = raw.postprocess()

        # Resize the image to 1920x1440
        resized_rgb = resize(rgb, (1440, 1920))

        # Crop unimportant segments of images
        if filename == "image_3.dng": 
            cropped_rgb = resized_rgb[:-360, :]
        elif filename == "image_2.dng":
            cropped_rgb = resized_rgb[180:-180, :]   
        else: 
            cropped_rgb = resized_rgb[360:, :]  

        # Convert the image to uint8
        uint8_rgb = (cropped_rgb * 255).astype(np.uint8)

        # Save the image as .png in the png directory
        imageio.imsave(os.path.join(png_dir, os.path.splitext(filename)[0] + '.png'), uint8_rgb)

# Print the list of .png files in the png directory, their new file size in MB, and resolution
for filename in os.listdir(png_dir):
    if filename.endswith('.png'):
        print(filename, '|', os.path.getsize(os.path.join(png_dir, filename))/1e6, 'MB', '|', imageio.imread(os.path.join(png_dir, filename)).shape)