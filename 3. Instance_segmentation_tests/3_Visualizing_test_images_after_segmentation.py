#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from pathlib import Path
import yaml
import cv2  # OpenCV is used to save the annotated image
from ultralytics import YOLO

# --- CONFIGURATION: PLEASE EDIT THESE PATHS ---

# 1. Path to the model weights you want to use for visualization
#    We'll use the 'best.pt' from the first initialization as an example.
MODEL_PATH = Path('/work3/msam/Thesis/segmentation/multiple_init_results/init_0/init_0/weights/best.pt')

# 2. Path to your original dataset.yaml file
YAML_PATH = Path('/work3/msam/Thesis/segmentation/yolodataset_seg/dataset.yaml')

# 3. Path to the directory where your test images are stored
#    The script will find this automatically from the YAML, but you can override it here.
TEST_IMAGE_DIR = None 

# 4. List of specific image filenames you want to visualize
#    These should be the names of files inside your test image directory.
#    I've added a few examples based on your description.
IMAGES_TO_VISUALIZE = [
    # --- Examples from the "Separated" set (e.g., numbers 1-40) ---
    "group_10/00020.png", # Example of a separated image
    "group_10/00010.png", # Another example of a separated image
    "group_21/00020.png",
    # --- Examples from the "Occluded/Touched" set (e.g., numbers 41-60) ---
    "group_20/00051.png", # Example of a touched/occluded image
    "group_21/00052.png",
    "group_20/00054.png", # Example of a touched/occluded image
    "group_21/00057.png",
    "group_20/00059.png", # Example of a touched/occluded image
    "group_21/00060.png"
    # Another example of a touched/occluded image
]

# 5. Directory to save the output images
OUTPUT_DIR = Path('/work3/msam/Thesis/segmentation/visualization_outputs')

# 6. Confidence threshold for predictions (0.0 to 1.0)
#    Only predictions with a score higher than this will be shown.
CONF_THRESHOLD = 0.5

# --- END OF CONFIGURATION ---


def main():
    """
    Main function to load the model, run predictions, and save visualizations.
    """
    print("🚀 Starting prediction visualization script...")

    # --- 1. Setup and Sanity Checks ---
    if not MODEL_PATH.exists():
        print(f"❌ ERROR: Model file not found at: {MODEL_PATH}")
        return

    if not YAML_PATH.exists():
        print(f"❌ ERROR: YAML file not found at: {YAML_PATH}")
        return

    # Create the output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ Output directory is ready at: {OUTPUT_DIR}")

    # Determine the test image directory from the YAML file
    global TEST_IMAGE_DIR
    if TEST_IMAGE_DIR is None:
        with open(YAML_PATH, 'r') as f:
            data_config = yaml.safe_load(f)
        TEST_IMAGE_DIR = YAML_PATH.parent / Path(data_config['test'])
        print(f"ℹ️  Test image directory identified from YAML: {TEST_IMAGE_DIR}")

    # --- 2. Load the YOLO Model ---
    print(f"\nLoading model from: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ ERROR: Failed to load the model. {e}")
        return

    # --- 3. Process and Visualize Each Image ---
    print("\nProcessing images for visualization...")
    for image_name in IMAGES_TO_VISUALIZE:
        image_path = TEST_IMAGE_DIR / image_name
        
        if not image_path.exists():
            print(f"⚠️ WARNING: Image not found, skipping: {image_path}")
            continue

        print(f"  -> Predicting on '{image_name}'...")

        # Run prediction
        try:
            results = model.predict(
                source=image_path,
                conf=CONF_THRESHOLD,
                verbose=False  # Keep the console output clean
            )
            
            # The result object has a 'plot()' method that draws annotations
            # It returns a NumPy array of the image (in BGR format for OpenCV)
            annotated_image = results[0].plot()
            
            # Create a descriptive output filename
            output_filename = OUTPUT_DIR / f"{image_path.stem}_prediction.png"
            
            # Save the annotated image using OpenCV
            cv2.imwrite(str(output_filename), annotated_image)
            
            print(f"     ✅ Saved prediction to: {output_filename}")

        except Exception as e:
            print(f"     ❌ ERROR: Failed to process {image_name}. {e}")

    print("\n🎉 Visualization complete!")


if __name__ == "__main__":
    # Ensure you have opencv-python installed:
    # pip install opencv-python
    main()

