#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats #calculating CI
from ultralytics import YOLO
import yaml
from datetime import datetime
import re # For parsing filenames

### Class names loading fuction
def load_class_names(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        return data.get('names', [])

### Model evaluation function (WITH THE FINAL FIX)
def evaluate_model(model_path, test_data_path):
    model = YOLO(model_path)
    
    # Run validation, explicitly telling it to use the 'test' split
    results = model.val(
        data=test_data_path,
        conf=0.5,
        iou=0.9,
        verbose=False,
        split='test'  # <-- THIS IS THE FINAL FIX
    )
    
    metrics = {}
    if hasattr(results, 'box'):
        num_classes = len(results.box.ap_class_index)
        
        # Gather metrics for each class
        for i in range(num_classes):
            p, r, ap50, ap = results.box.class_result(i)
            class_name = f"class_{results.box.ap_class_index[i]}"
            metrics[class_name] = {
                'precision': float(p),
                'recall': float(r),
                'map50': float(ap50),
                'map50_95': float(ap)
            }
        
        # Also store mean metrics for all classes
        metrics['all_classes'] = {
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'map50': float(results.box.map50),
            'map50_95': float(results.box.map)
        }
    
    return metrics

### function to calculate the confidence intervals (95%)
def calculate_confidence_interval(data, confidence=0.95):
    array = np.array(data)
    mean = np.mean(array)
    ci = stats.t.interval(
        confidence,
        len(array) - 1,
        loc=mean,
        scale=stats.sem(array)
    )
    return {
        'mean': mean,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'std': np.std(array),
        'min': np.min(array),
        'max': np.max(array)
    }

def perform_evaluation_on_subset(
    subset_name, image_paths, original_yaml_path, class_names, base_dir, output_dir, timestamp
):
    """
    Runs the full evaluation pipeline for a specific subset of images.
    """
    # (No changes needed in the first part of the function)
    print(f"\n{'='*80}")
    print(f"STARTING EVALUATION FOR: '{subset_name.upper()}' SUBSET ({len(image_paths)} images)")
    print(f"{'='*80}")

    # --- 1. Create temporary data files for the subset ---
    temp_list_file = output_dir / f"temp_{subset_name}_imagelist.txt"
    with open(temp_list_file, 'w') as f:
        for path in image_paths:
            f.write(f"{path}\n")

    with open(original_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)

    if 'train' not in data_config:
        placeholder_path = Path(data_config['path']) / data_config.get('val', data_config['test'])
        data_config['train'] = str(placeholder_path)

    if 'val' not in data_config:
        placeholder_path = Path(data_config['path']) / data_config['test']
        data_config['val'] = str(placeholder_path)
    
    data_config['test'] = str(temp_list_file)
    
    temp_yaml_path = output_dir / f"temp_{subset_name}_config.yaml"
    with open(temp_yaml_path, 'w') as f:
        yaml.dump(data_config, f)

    # --- 2. Evaluate each model initialization ---
    all_metrics = {}
    for init in range(10): 
        model_path = base_dir / f'init_{init}/init_{init}/weights/best.pt'
        print(f"\nProcessing initialization {init} for '{subset_name}' subset...")
        
        if model_path.exists():
            try:
                metrics = evaluate_model(str(model_path), str(temp_yaml_path))
                for class_key, class_metrics in metrics.items():
                    if class_key not in all_metrics:
                        all_metrics[class_key] = {m: [] for m in class_metrics.keys()}
                    for metric, value in class_metrics.items():
                        all_metrics[class_key][metric].append(value)
            except Exception as e:
                print(f"Error processing initialization {init} for subset '{subset_name}': {e}")
        else:
            print(f"Warning: Model not found at {model_path}")

    # --- 3. Calculate statistics and save results ---
    results = []
    for class_key, metrics_data in all_metrics.items():
        for metric_name, values in metrics_data.items():
            if values:
                stats_result = calculate_confidence_interval(values)
                results.append({
                    'class': class_key, 'metric': metric_name, **stats_result
                })
    
    if not results:
        print(f"\nNo results generated for subset '{subset_name}'. Skipping CSV creation.")
        os.remove(temp_list_file)
        os.remove(temp_yaml_path)
        return

    results_df = pd.DataFrame(results)
    
    if class_names:
        # --- THIS IS THE CORRECTED SECTION ---
        # It now correctly iterates over the dictionary items from your YAML
        class_map = {f'class_{i}': name for i, name in class_names.items()}
        class_map['all_classes'] = 'Overall' 
        results_df['class'] = results_df['class'].replace(class_map)
    
    results_csv_path = output_dir / f'{subset_name}_metrics_{timestamp}.csv'
    results_df.to_csv(results_csv_path, index=False, float_format='%.4f')
    
    print("\n" + f"Final Results for '{subset_name.upper()}' Subset".center(80, "-"))
    map50_95_results = results_df[results_df['metric'] == 'map50_95']
    map50_95_results['class'] = pd.Categorical(map50_95_results['class'], ['Overall'] + [c for c in sorted(map50_95_results['class'].unique()) if c != 'Overall'])
    map50_95_results = map50_95_results.sort_values(by='class')
    
    for _, row in map50_95_results.iterrows():
        # Added a check for NaN values to prevent printing errors for the CI
        if pd.notna(row['mean']) and pd.notna(row['ci_upper']) and pd.notna(row['ci_lower']):
            se = (row['ci_upper'] - row['ci_lower']) / (2 * 2.262)
            print(f"{row['class']:<15} | mAP50-95: {row['mean']*100:.2f} ± {se*100:.2f}")
        else:
            print(f"{row['class']:<15} | mAP50-95: N/A (not enough data)")

    print("-" * 80)
    print(f"Full results for '{subset_name}' saved to: {results_csv_path}")

    # --- 4. Clean up temporary files ---
    os.remove(temp_list_file)
    os.remove(temp_yaml_path)

def main():
    # --- Basic Setup ---
    base_dir = Path('/work3/msam/Thesis/segmentation/multiple_init_results')
    original_yaml_path = Path('/work3/msam/Thesis/segmentation/yolodataset_seg/dataset.yaml')
    output_dir = Path('/work3/msam/Thesis/segmentation/test_evaluations_subsets')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # --- Load Class Names ---
    try:
        class_names = load_class_names(original_yaml_path)
        print(f"Loaded class names: {class_names}")
    except Exception as e:
        print(f"Could not load class names: {e}")
        class_names = []

    # --- Identify Test Image Directory ---
    with open(original_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    test_image_dir = original_yaml_path.parent / Path(data_config['test'])
    print(f"Test image directory found at: {test_image_dir}")

    # --- Create Image Subsets ---
    separated_images = []
    occluded_images = []
    
    all_images = list(test_image_dir.glob('**/*.[jp][pn]g')) 
    
    print(f"\nFound {len(all_images)} total image files across all sub-directories. Now sorting...")

    for img_path in all_images:
        try:
            filename_part = img_path.stem.split('_')[-1]
            img_num = int(filename_part)
            
            if 1 <= img_num <= 40:
                separated_images.append(img_path)
            elif 41 <= img_num <= 60:
                occluded_images.append(img_path)

        except (ValueError, IndexError):
            print(f"  - Could not parse number from '{img_path.name}', skipping.")
            continue
            
    print(f"\nIdentified {len(separated_images)} 'separated' images (e.g., 1-40).")
    print(f"Identified {len(occluded_images)} 'occluded' images (e.g., 41-60).")

    # --- Run Evaluation for Each Subset ---
    subsets_to_evaluate = {
        "separated": separated_images,
        "occluded": occluded_images
    }

    for name, paths in subsets_to_evaluate.items():
        if paths: 
            perform_evaluation_on_subset(
                subset_name=name,
                image_paths=paths,
                original_yaml_path=original_yaml_path,
                class_names=class_names,
                base_dir=base_dir,
                output_dir=output_dir,
                timestamp=timestamp
            )
        else:
            print(f"\nSkipping evaluation for '{name}' subset as no images were found.")

if __name__ == "__main__":
    main()

