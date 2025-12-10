#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader
import os
from tqdm import tqdm 
import numpy as np # For intermediate calculations if preferred, though torch can handle it


# --- Configuration 
# defining the dataset and data loader

dataset_path = '/work3/msam/Thesis/autofish/metric_learning_instance_split/calibration_folder' 
N_CHANNELS = 3  
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count() # Use available CPU cores for loading

# --- Dataset and DataLoader ---
# Only apply ToTensor() to scale images to [0.0, 1.0]
# If your images are not already in a standard size, you might consider
# adding a transforms.Resize() here if you want the stats for a fixed size.
# However, for true dataset stats, it's often done on original aspect ratio images
# scaled to [0,1].
# For now, let's assume ToTensor() is sufficient.


## strategy to preserve the fish details without cropping the body parts during resizeing
## in here the padding will be applied to the sides by preserving the details
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

# The gourps are made per each fish id---------------
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

