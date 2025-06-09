#!/usr/bin/env python
# coding: utf-8

# ## Dataset rearrangement

# In[ ]:


import os
import re
import shutil

def organize_images_by_id(source_folder_path, output_base_folder_path):
    """
    Organizes images from a source folder into subfolders named by IDs
    extracted from the image filenames.

    Args:
        source_folder_path (str): Path to the folder containing the images
                                 (e.g., your 'all_images' folder).
        output_base_folder_path (str): Path to the directory where the new
                                     ID-specific subfolders will be created.
    """
    print(f"Source folder: {source_folder_path}")
    print(f"Output base folder: {output_base_folder_path}")

    # Create the base output directory if it doesn't exist
    if not os.path.exists(output_base_folder_path):
        os.makedirs(output_base_folder_path)
        print(f"Created output base folder: {output_base_folder_path}")

    # Regex to extract the ID (e.g., '316' from '..._id316.png')
    # This pattern looks for '_id' followed by one or more digits, then '.png'
    # It captures only the digits.
    id_pattern = re.compile(r"_id(\d+)\.png$", re.IGNORECASE) # Added IGNORECASE for .PNG/.png

    files_processed = 0
    files_moved = 0
    files_with_no_id_match = []

    for filename in os.listdir(source_folder_path):
        files_processed += 1
        match = id_pattern.search(filename)

        if match:
            fish_id = match.group(1)  # Get the captured digits (the ID)

            # Create the ID-specific subfolder in the output directory
            id_specific_folder_path = os.path.join(output_base_folder_path, fish_id)
            if not os.path.exists(id_specific_folder_path):
                os.makedirs(id_specific_folder_path)
                # print(f"  Created ID folder: {id_specific_folder_path}")

            # Define source and destination paths for the file
            source_file_path = os.path.join(source_folder_path, filename)
            destination_file_path = os.path.join(id_specific_folder_path, filename)

            # Move the file
            try:
                shutil.move(source_file_path, destination_file_path)
                # print(f"  Moved '{filename}' to '{id_specific_folder_path}'")
                files_moved += 1
            except Exception as e:
                print(f"  ERROR moving '{filename}': {e}")
        else:
            # print(f"  Filename '{filename}' does not match ID pattern. Skipping.")
            files_with_no_id_match.append(filename)

    print(f"\n--- Summary ---")
    print(f"Total files processed in source folder: {files_processed}")
    print(f"Files successfully moved: {files_moved}")
    if files_with_no_id_match:
        print(f"Files that did not match the ID pattern ({len(files_with_no_id_match)}):")
        for skipped_file in files_with_no_id_match:
            print(f"  - {skipped_file}")
    else:
        print("All processable files matched the ID pattern.")
    print("-----------------")

# --- How to use the script ---
if __name__ == "__main__":
    # IMPORTANT: SET THESE PATHS CORRECTLY!

    # Path to your folder that currently contains all the images (e.g., 'all_images')
    # As per your image, this folder is one level up from where the files are.
    # If your 'all_images' folder is at '/work3/msam/Thesis/autofish/metric_learning_instance_split/train_is/all_images'
    current_images_folder = "/work3/msam/Thesis/autofish/metric_learning_instance_split/train_is" # EXAMPLE: "/work3/msam/Thesis/autofish/dataset_flat/all_images"

    # Path where you want the new ID-organized folders to be created
    # For example, this could be a new folder like "train_is_organized_by_id"
    # at the same level as your 'train_is' folder.
    output_organized_folder = "/work3/msam/Thesis/autofish/metric_learning_instance_split/calibration_folder" # EXAMPLE: "/work3/msam/Thesis/autofish/dataset_organized_by_id"

    # --- Safety Check ---
    # Check if the paths have been changed from the placeholder examples
    if current_images_folder == "/path/to/your/all_images" or \
       output_organized_folder == "/path/to/your/new_organized_dataset_folder":
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! PLEASE UPDATE 'current_images_folder' and               !!!")
        print("!!! 'output_organized_folder' variables in the script       !!!")
        print("!!! with the correct paths for your system before running.  !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        # Confirm before running - this is a destructive operation (moves files)
        print(f"\nAbout to process files from: '{current_images_folder}'")
        print(f"They will be MOVED into ID-specific subfolders inside: '{output_organized_folder}'")
        
        user_confirmation = input("Are you sure you want to proceed? (yes/no): ").strip().lower()

        if user_confirmation == 'yes':
            print("\nStarting organization process...")
            organize_images_by_id(current_images_folder, output_organized_folder)
            print("Organization process finished.")
        else:
            print("Operation cancelled by the user.")


# ## calcaulting the mean and std of the autofish datset

# In[ ]:


import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader
import os
from tqdm import tqdm # For a nice progress bar
import numpy as np # For intermediate calculations if preferred, though torch can handle it

# ----------------------------------------
# --- Configuration 
#deifining the dataset and data loader

#-----------------------------------------
dataset_path = '/work3/msam/Thesis/autofish/metric_learning_instance_split/calibration_folder' 
N_CHANNELS = 3 # Assuming color images (RGB)
BATCH_SIZE = 16 # Adjust based on your memory
NUM_WORKERS = os.cpu_count() # Use available CPU cores for loading

# --- Dataset and DataLoader ---
# Only apply ToTensor() to scale images to [0.0, 1.0]
# If your images are not already in a standard size, you might consider
# adding a transforms.Resize() here if you want the stats for a fixed size.
# However, for true dataset stats, it's often done on original aspect ratio images
# scaled to [0,1].
# For now, let's assume ToTensor() is sufficient.


## strategy to preserve the fish details without cropping the body parts during resizeing
## in here the padding will be appied to the sides by preserving the details
class ResizeAndPadToSquare:
    def __init__(self, output_size_square, fill_color=(0, 0, 0)):
        """
        Args:
            output_size_square (int): The desired height and width of the square output image.
            fill_color (tuple): RGB tuple for the padding color. Default is black.
        """
        assert isinstance(output_size_square, int)
        self.output_size = output_size_square
        self.fill_color = fill_color

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to resize and pad.
        Returns:
            PIL Image: Resized and padded image.
        """
        # Get original image dimensions
        original_w, original_h = img.size

        # Determine new dimensions to fit within output_size_square while maintaining aspect ratio
        if original_w > original_h:
            # Image is wider than it is tall
            new_w = self.output_size
            new_h = int(self.output_size * (original_h / original_w))
        elif original_h > original_w:
            # Image is taller than it is wide
            new_h = self.output_size
            new_w = int(self.output_size * (original_w / original_h))
        else:
            # Image is already square
            new_w = self.output_size
            new_h = self.output_size

        # Resize the image using functional transforms (expects H, W order for size)
        resized_img = TF.resize(img, (new_h, new_w))

        # Calculate padding
        # The padding tuple is (left, top, right, bottom)
        pad_left = (self.output_size - new_w) // 2
        pad_right = self.output_size - new_w - pad_left
        pad_top = (self.output_size - new_h) // 2
        pad_bottom = self.output_size - new_h - pad_top

        padding = (pad_left, pad_top, pad_right, pad_bottom)

        # Pad the image
        padded_img = TF.pad(resized_img, padding, fill=self.fill_color, padding_mode='constant')

        return padded_img



FINAL_SQUARE_SIZE = 224

transform_for_stats = transforms.Compose([
    ResizeAndPadToSquare(FINAL_SQUARE_SIZE, fill_color=(0,0,0)), # Our custom transform
    transforms.ToTensor()
])

# Assuming your dataset is structured for ImageFolder
# (e.g., root_folder/train/class1/img.jpg, root_folder/train/class2/img.jpg)
# If you only have one folder of images without subfolders for classes,
# you might need a custom Dataset or structure it like:
# root_folder/all_fish/your_images.jpg
try:
    # If your dataset has a 'train' subfolder, use that.
    # Otherwise, point directly to the folder containing class subfolders,
    # or the folder containing all images if using a custom dataset.
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform_for_stats)
except Exception as e:
    print(f"Error loading dataset with ImageFolder: {e}")
    print("Please ensure your dataset_path is correct and images are organized,")
    print("e.g., dataset_root_folder/class_name/image.jpg")
    print("If you have a custom Dataset class, replace 'datasets.ImageFolder' accordingly.")
    exit()

# If your dataset is very large, you might calculate stats on a representative subset
# For example, if you have separate train/val/test, calculate on the training set.
dataloader_for_stats = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print(f"Calculating mean and std for {len(full_dataset)} images...")

# Variables to store sum of pixel values and sum of squared pixel values
# Initialize for N_CHANNELS
sum_pixels = torch.zeros(N_CHANNELS, dtype=torch.float64)  # Use float64 for precision
sum_sq_pixels = torch.zeros(N_CHANNELS, dtype=torch.float64)
num_total_pixels = 0

# Iterate through the DataLoader
for inputs, _ in tqdm(dataloader_for_stats):
    # inputs shape: (batch_size, N_CHANNELS, height, width)

    # Flatten height and width dimensions for each channel
    # For each channel, sum all pixel values in the batch
    # inputs.sum(dim=[0, 2, 3]) would sum across batch, height, and width for each channel if not careful
    # We want to sum pixel values first, then sum those sums.

    # Permute to (N_CHANNELS, batch_size, height, width) then reshape
    # Or iterate per channel if clearer
    for i in range(N_CHANNELS):
        # Sum of pixels for channel i in the current batch
        sum_pixels[i] += torch.sum(inputs[:, i, :, :])
        # Sum of squared pixels for channel i in the current batch
        sum_sq_pixels[i] += torch.sum(inputs[:, i, :, :] ** 2)

    # Count total pixels processed
    # batch_size * height * width
    num_total_pixels += inputs.size(0) * inputs.size(2) * inputs.size(3)


# Calculate mean for each channel
mean = sum_pixels / num_total_pixels

# Calculate variance for each channel: E[X^2] - (E[X])^2
# E[X^2] = sum_sq_pixels / num_total_pixels
# (E[X])^2 = mean^2
variance = (sum_sq_pixels / num_total_pixels) - (mean ** 2)

# Ensure variance is not negative due to floating point inaccuracies for very small variances
variance[variance < 0] = 0 

std = torch.sqrt(variance)

print(f"\nCalculated statistics for {N_CHANNELS} channels:")
print(f"Mean: {mean.tolist()}") # .tolist() for cleaner printing
print(f"Std:  {std.tolist()}")

# Example usage for transforms.Normalize
# normalize_transform = transforms.Normalize(mean=mean.tolist(), std=std.tolist())
# print(f"\nTo use in transforms.Normalize:\nnormalize = transforms.Normalize(mean={mean.tolist()}, std={std.tolist()})")


# ## visualizing the distribution

# In[ ]:


import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from PIL import Image

# --- Custom Transform Definition ---
class ResizeAndPadToSquare:
    def __init__(self, output_size_square, fill_color=(0, 0, 0)):
        assert isinstance(output_size_square, int)
        self.output_size = output_size_square
        self.fill_color = fill_color

    def __call__(self, img):
        original_w, original_h = img.size
        if original_w > original_h:
            new_w = self.output_size
            new_h = int(self.output_size * (original_h / original_w))
        elif original_h > original_w:
            new_h = self.output_size
            new_w = int(self.output_size * (original_w / original_h))
        else:
            new_w = self.output_size
            new_h = self.output_size
        
        resized_img = TF.resize(img, (new_h, new_w)) # TF.resize expects (h, w)

        pad_left = (self.output_size - new_w) // 2
        pad_right = self.output_size - new_w - pad_left
        pad_top = (self.output_size - new_h) // 2
        pad_bottom = self.output_size - new_h - pad_top
        padding = (pad_left, pad_top, pad_right, pad_bottom)
        
        padded_img = TF.pad(resized_img, padding, fill=self.fill_color, padding_mode='constant')
        return padded_img

# --- Configuration ---
dataset_path = '/work3/msam/Thesis/autofish/metric_learning_instance_split/calibration_folder'
N_CHANNELS = 3
BATCH_SIZE = 32  # Can be larger for visualization if memory allows
NUM_WORKERS = os.cpu_count()
FINAL_SQUARE_SIZE = 224 # Use the same size as your mean/std calculation and intended model input

# Your calculated statistics
calculated_means = [0.049543730340171566, 0.05032391348629852, 0.05353340628871572]
calculated_stds = [0.1370093056773738, 0.13629327765904778, 0.14115511484815407]

# --- Transforms for Visualization ---
# Should be the same as used for mean/std calculation (before normalization)
transform_for_viz = transforms.Compose([
    ResizeAndPadToSquare(FINAL_SQUARE_SIZE, fill_color=(0,0,0)),
    transforms.ToTensor()
])

# --- Dataset and DataLoader ---
print(f"Loading dataset from: {dataset_path}")
try:
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform_for_viz)
    print(f"Found {len(full_dataset)} images.")
except Exception as e:
    print(f"Error loading dataset with ImageFolder: {e}")
    print("Please ensure your dataset_path is correct and images are organized,")
    print("e.g., into a dummy subfolder if no classes, or class-specific subfolders.")
    exit()

# We'll visualize a subset, e.g., first few batches, to avoid loading everything if dataset is huge
# Adjust num_batches_to_viz as needed
num_batches_to_viz = 10 # e.g., 10 batches * 32 images/batch = 320 images
images_to_viz_limit = BATCH_SIZE * num_batches_to_viz

dataloader_for_viz = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS) # Shuffle to get a varied sample

# --- Collect Pixel Values ---
# Lists to store pixel values for each channel
pixel_values_r = []
pixel_values_g = []
pixel_values_b = []

print(f"Collecting pixel values from a subset of images (up to ~{images_to_viz_limit} images)...")
images_processed_count = 0
for i, (inputs, _) in enumerate(dataloader_for_viz):
    if images_processed_count >= images_to_viz_limit:
        break
    
    # inputs shape: (batch_size, N_CHANNELS, height, width)
    # Values are already scaled to [0, 1] by ToTensor()
    
    # Channel 0 (Red)
    pixel_values_r.extend(inputs[:, 0, :, :].flatten().tolist())
    # Channel 1 (Green)
    pixel_values_g.extend(inputs[:, 1, :, :].flatten().tolist())
    # Channel 2 (Blue)
    pixel_values_b.extend(inputs[:, 2, :, :].flatten().tolist())
    
    images_processed_count += inputs.size(0)
    if i + 1 >= num_batches_to_viz: # Stop after num_batches_to_viz
        break

print(f"Collected pixel values from {images_processed_count} images.")

if images_processed_count == 0:
    print("No images were processed. Cannot generate plot. Check dataset and paths.")
    exit()

# --- Plotting ---
print("Generating histograms...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True) # sharey for easier comparison

channel_names = ['Red Channel', 'Green Channel', 'Blue Channel']
all_pixel_values = [pixel_values_r, pixel_values_g, pixel_values_b]

for i in range(N_CHANNELS):
    ax = axes[i]
    ax.hist(all_pixel_values[i], bins=100, range=(0, 1), color=['red', 'green', 'blue'][i], alpha=0.7, density=True)
    ax.axvline(calculated_means[i], color='k', linestyle='dashed', linewidth=2, label=f'Mean: {calculated_means[i]:.4f}')
    
    # Optional: Add text for std dev
    # ax.text(0.95, 0.90, f'Std: {calculated_stds[i]:.4f}', 
    #         horizontalalignment='right', verticalalignment='top', 
    #         transform=ax.transAxes, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    ax.set_title(channel_names[i])
    ax.set_xlabel("Pixel Value (0-1)")
    if i == 0:
        ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.7)

fig.suptitle(f'Pixel Value Distribution (from {images_processed_count} images)\nDataset: {os.path.basename(dataset_path)}', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle

# Save the plot
plot_filename = "pixel_distribution_histogram.png"
try:
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")
except Exception as e:
    print(f"Error saving plot: {e}")

# Show the plot (optional, comment out if running in a non-GUI environment)
plt.show()

print("Done.")


# In[2]:


import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# DataLoader might not be needed if we fetch directly from dataset for viz
# from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from PIL import Image
import math
import random # To shuffle indices if we want random samples
import numpy as np

# --- Custom Transform Definition (same as before) ---
class ResizeAndPadToSquare:
    def __init__(self, output_size_square, fill_color=(0, 0, 0)):
        assert isinstance(output_size_square, int)
        self.output_size = output_size_square
        self.fill_color = fill_color

    def __call__(self, img):
        original_w, original_h = img.size
        if original_w > original_h:
            new_w = self.output_size
            new_h = int(self.output_size * (original_h / original_w))
        elif original_h > original_w:
            new_h = self.output_size
            new_w = int(self.output_size * (original_w / original_h))
        else:
            new_w = self.output_size
            new_h = self.output_size
        
        resized_img = TF.resize(img, (new_h, new_w))

        pad_left = (self.output_size - new_w) // 2
        pad_right = self.output_size - new_w - pad_left
        pad_top = (self.output_size - new_h) // 2
        pad_bottom = self.output_size - new_h - pad_top
        padding = (pad_left, pad_top, pad_right, pad_bottom)
        
        padded_img = TF.pad(resized_img, padding, fill=self.fill_color, padding_mode='constant')
        return padded_img

# --- Configuration ---
dataset_path = '/work3/msam/Thesis/autofish/metric_learning_instance_split/calibration_folder'
FINAL_SQUARE_SIZE = 224
NUM_IMAGES_TO_SHOW = 8
# NUM_WORKERS is not strictly needed if not using DataLoader, but good for ImageFolder if it uses it internally for something
NUM_WORKERS = os.cpu_count() 

# --- Transforms for Visualization ---
transform_for_image_viz = ResizeAndPadToSquare(FINAL_SQUARE_SIZE, fill_color=(0,0,0))

# --- Dataset ---
print(f"Loading dataset from: {dataset_path}")
try:
    viz_dataset = datasets.ImageFolder(root=dataset_path, transform=transform_for_image_viz)
    print(f"Found {len(viz_dataset)} images.")
    if len(viz_dataset) == 0:
        print("Dataset is empty. Please check the path and structure.")
        exit()
except Exception as e:
    print(f"Error loading dataset with ImageFolder: {e}")
    # ... (rest of your error handling) ...
    exit()

# --- Get a Sample of Images Directly from Dataset ---
images_to_plot = []
actual_num_images_shown = 0

if len(viz_dataset) > 0:
    num_to_fetch = min(NUM_IMAGES_TO_SHOW, len(viz_dataset))
    
    # Get random indices for a varied sample
    indices_to_fetch = random.sample(range(len(viz_dataset)), num_to_fetch)
    
    print(f"\nFetching {num_to_fetch} sample images directly from dataset...")
    for i in indices_to_fetch:
        try:
            # When accessing directly, dataset[i] returns (image_PIL, label)
            img_pil, _label = viz_dataset[i] 
            images_to_plot.append(img_pil)
        except Exception as e:
            print(f"Error fetching image at index {i}: {e}")
            continue # Skip this image if there's an error
    actual_num_images_shown = len(images_to_plot)
else:
    print("Dataset is empty, no images to fetch.")

if actual_num_images_shown > 0:
    # --- Plotting the Images ---
    print(f"Displaying {actual_num_images_shown} sample images after ResizeAndPadToSquare ({FINAL_SQUARE_SIZE}x{FINAL_SQUARE_SIZE}):")

    cols = 4
    rows = math.ceil(actual_num_images_shown / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if actual_num_images_shown == 1: # If only one image, axes is not an array
        axes = [axes] 
    axes = np.array(axes).flatten() # Ensure axes is always a flat NumPy array for consistent indexing


    for i, img_pil in enumerate(images_to_plot):
        if i < len(axes):
            ax = axes[i]
            ax.imshow(img_pil)
            ax.set_title(f"Sample {i+1}")
            ax.axis('off')

    for j in range(actual_num_images_shown, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"Images after ResizeAndPadToSquare ({FINAL_SQUARE_SIZE}x{FINAL_SQUARE_SIZE})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plot_filename = "padded_image_visualization.png"
    try:
        plt.savefig(plot_filename)
        print(f"Plot saved as {plot_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    plt.show()
else:
    print("No images were successfully fetched to display.")

print("Done.")


# In[ ]:




