__author__ = 'Taylor Bybee'
__copyright__ = 'Copyright (C) 2023-2024 Taylor Bybee'

# Imports
import numpy as np
import argparse
import math
import cv2
import os
import matplotlib.pyplot as plt


def ParseArgs():
    """
    This function parses arguments to the program.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    args = parser.parse_args()
    return args


def ComputeRmse(img1, img2):
    i1 = img1.flatten().astype(float)
    i2 = img2.flatten().astype(float)
    d = i1 - i2
    return math.sqrt(np.mean(np.multiply(d, d)))


def main():
    # Parse Arguments
    args = ParseArgs()

    # Load Raw Image
    original_img = cv2.imread(args.image)
    original_img_size = os.path.getsize(args.image)
    print(f'Original image size: {original_img_size} bytes with dimensions {original_img.shape}.')

    # Write out various compression levels
    jpeg_quality_levels = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    image_name = args.image.split("/")[-1].replace(".png", "")
    folder_path = 'photos/compressed_images/'  + image_name + '/'

    for jql in jpeg_quality_levels:
        # Create filename
        fn = folder_path + image_name + '_' + str(jql) + '.jpg'

        # Write file to disk with the specified compression level
        cv2.imwrite(fn, original_img, [cv2.IMWRITE_JPEG_QUALITY, jql])

        # Compute some metrics, print info to console
        read_img = cv2.imread(fn)
        rmse = ComputeRmse(read_img, original_img)
        compressed_img_size = os.path.getsize(fn)
        r = compressed_img_size / original_img_size
        print(f'JPEG Quality: {jql}. Size on disk: {compressed_img_size} bytes. Compression ratio: {r}. Pixel RMSE: {rmse}.')

    # Plot the compression ratio by compression and a straight line for 30 quality level
    plt.plot(jpeg_quality_levels, [os.path.getsize(folder_path + image_name + '_' + str(jql) + '.jpg')/original_img_size for jql in jpeg_quality_levels], label='Compression Ratio')
    plt.axvline(x=30, color='r', linestyle='-', label='Quality Level 30')
    plt.xlabel('JPEG Quality Level')
    plt.ylabel('Compression Ratio')
    plt.title('Compression Ratio by Compression Level')
    plt.legend()
    plt.savefig('photos/plot_grids/' + image_name + '_compression_ratio_plot.jpg')
    plt.show()

    # Plot the image size by compression and a straight line for 0.55 MB
    plt.plot(jpeg_quality_levels, [os.path.getsize(folder_path + image_name + '_' + str(jql) + '.jpg')/1e6 for jql in jpeg_quality_levels], label='Image Size (MB)')
    plt.axhline(y=0.55, color='r', linestyle='-', label='0.55 MB')
    plt.xlabel('JPEG Quality Level')
    plt.ylabel('Image Size (MB)')
    plt.title('Image Size by Compression Level')
    plt.legend()
    plt.savefig('photos/plot_grids/' + image_name + '_size_plot.jpg')
    plt.show()

    # Plot the image RMSE by compression and a straight vertical line for quality level 30
    plt.plot(jpeg_quality_levels, [ComputeRmse(cv2.imread(folder_path + image_name + '_' + str(jql) + '.jpg'), original_img) for jql in jpeg_quality_levels], label='Pixel RMSE')
    plt.axvline(x=30, color='r', linestyle='-', label='Quality Level 30')
    plt.xlabel('JPEG Quality Level')
    plt.ylabel('Pixel RMSE')
    plt.title('Pixel RMSE by Compression Level')
    plt.legend()
    plt.savefig('photos/plot_grids/' + image_name + '_rmse_plot.jpg')
    plt.show()

    # Create a 6x2 subplot of compressed images
    fig, axs = plt.subplots(6, 2, figsize=(10, 24))

    # Show the original image in the first subplot
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)  # Convert color space for matplotlib
    axs[0, 0].imshow(original_img_rgb)
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')  # Hide axis

    # Show the compressed images in the remaining subplots
    for i, jql in enumerate(jpeg_quality_levels, start=1):
        fn = folder_path + image_name + '_' + str(jql) + '.jpg'
        read_img = cv2.imread(fn)
        read_img = cv2.cvtColor(read_img, cv2.COLOR_BGR2RGB)  # Convert color space for matplotlib
        axs[i//2, i%2].imshow(read_img)
        axs[i//2, i%2].set_title(f'JPEG Quality: {jql}')
        axs[i//2, i%2].axis('off')  # Hide axis

    # Save plot to plot_grids folder and name it after the original image
    plot_fn = image_name + '_plot.jpg'
    plt.savefig('photos/plot_grids/' + plot_fn)

    plt.show()


# Entry Point
if __name__ == '__main__':
    main()