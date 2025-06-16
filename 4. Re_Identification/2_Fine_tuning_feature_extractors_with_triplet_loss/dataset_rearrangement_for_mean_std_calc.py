#!/usr/bin/env python
# coding: utf-8

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

