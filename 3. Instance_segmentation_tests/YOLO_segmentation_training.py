#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from ultralytics import YOLO
import torch
import os
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime

# Paths
data_path = "/work3/msam/Thesis/segmentation/yolodataset_seg/dataset.yaml"
model_path = "yolo11l-seg.pt"
output_dir = "/work3/msam/Thesis/segmentation/multiple_init_results"
os.makedirs(output_dir, exist_ok=True)

# Training parameters
hyperparameters = {
    "epochs": 300,
    "batch_size": 32,
    "img_size": 640,
    "device": 0,
    "optimizer": "AdamW",
    "n_initializations": 10
}

def train_single_initialization(model_path: str, data_path: str, output_dir: str, seed: int) -> Dict:
    """
    Train a single initialization
    """
    # Create directory for this initialization
    init_dir = os.path.join(output_dir, f'init_{seed}')
    os.makedirs(init_dir, exist_ok=True)
    
    # Log file for this initialization
    log_file = os.path.join(init_dir, 'training_log.txt')
    with open(log_file, 'w') as f:
        f.write(f"Starting initialization {seed} at {datetime.now()}\n")
    
    # Set random seed
    torch.manual_seed(seed)
    
    try:
        # Initialize model
        model = YOLO(model_path)
        
        # Train model
        results = model.train(
            data=data_path,
            epochs=hyperparameters["epochs"],
            imgsz=hyperparameters["img_size"],
            batch=hyperparameters["batch_size"],
            optimizer=hyperparameters["optimizer"],
            device=hyperparameters["device"],
            project=init_dir,
            name=f"init_{seed}",
            lr0=0.01,      # initial learning rate
            lrf=0.00001    # final learning rate
        )
        
        # Validate model
        val_results = model.val(
            data=data_path,
            imgsz=hyperparameters["img_size"],
            save_json=True,
            save_conf=True
        )
        
        # Store metrics
        metrics = {
            'initialization': seed,
            'mAP50': float(val_results.seg.map50),
            'mAP50-95': float(val_results.seg.map),
            'precision': float(val_results.seg.mp),
            'recall': float(val_results.seg.mr)
        }
        
        # Log completion
        with open(log_file, 'a') as f:
            f.write(f"Completed initialization {seed} at {datetime.now()}\n")
            f.write(f"Metrics: {metrics}\n")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
        return metrics
        
    except Exception as e:
        # Log error
        with open(log_file, 'a') as f:
            f.write(f"Error in initialization {seed}: {str(e)}\n")
        raise e

def calculate_statistics(results: List[Dict]) -> pd.DataFrame:
    """
    Calculate statistics across all initializations
    """
    df = pd.DataFrame(results)
    
    # Calculate statistics for each metric
    stats = []
    for metric in ['mAP50', 'mAP50-95', 'precision', 'recall']:
        values = df[metric].values
        stats.append({
            'metric': metric,
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'confidence_95': 1.96 * np.std(values) / np.sqrt(len(values))
        })
    
    return pd.DataFrame(stats)

def main():
    print("\nStarting Multiple Initialization Training")
    print("="*50)
    print(f"Training Parameters:")
    print(f"Initial LR: 0.01")
    print(f"Final LR: 0.00001")
    print(f"Epochs: {hyperparameters['epochs']}")
    print("="*50)
    
    # Store all results
    all_results = []
    
    # Run initializations
    for init in range(hyperparameters['n_initializations']):
        print(f"\nStarting initialization {init + 1}/{hyperparameters['n_initializations']}")
        
        try:
            # Train and get metrics
            metrics = train_single_initialization(model_path, data_path, output_dir, seed=init)
            all_results.append(metrics)
            
            # Save current results
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(os.path.join(output_dir, 'all_initializations.csv'), index=False)
            
            # Print current results
            print(f"\nResults from initialization {init + 1}:")
            for key, value in metrics.items():
                if key != 'initialization':
                    print(f"{key}: {value:.4f}")
            
        except Exception as e:
            print(f"Error in initialization {init}: {e}")
    
    # Calculate and save final statistics
    if all_results:
        print("\nCalculating final statistics...")
        stats_df = calculate_statistics(all_results)
        stats_df.to_csv(os.path.join(output_dir, 'final_statistics.csv'), index=False)
        
        # Print final results
        print("\nFinal Results:")
        print("-"*50)
        for _, row in stats_df.iterrows():
            print(f"\n{row['metric']}:")
            print(f"Mean: {row['mean']:.4f}")
            print(f"Std:  {row['std']:.4f}")
            print(f"95% CI: ±{row['confidence_95']:.4f}")
            print(f"Min:  {row['min']:.4f}")
            print(f"Max:  {row['max']:.4f}")

if __name__ == "__main__":
    main()

