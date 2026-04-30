import numpy as np
import matplotlib.pyplot as plt
import os
import random

# --- CONFIGURATION ---
# This should match the LAYOUT_NAME used in the generation scripts
LAYOUT_NAME = "tt_um_yen" 
BASE_OUT_DIR = f"outputs/{LAYOUT_NAME}"
TRAINING_DATASET = f"{BASE_OUT_DIR}/training_dataset"
CLEAN_DIR = f"{TRAINING_DATASET}/clean"
DIRTY_DIR = f"{TRAINING_DATASET}/dirty"
NUM_SAMPLES = 6 # 6 per row = 12 samples total

# Plot parameters corresponding to dataset generator
WINDOW_SIZE = 1600
CLIP_SIZE = 200
STRIDE = 400

# Calculate resolution automatically
RESOLUTION = WINDOW_SIZE / CLIP_SIZE

def visualize_dataset_matrices(clean_dir=CLEAN_DIR, dirty_dir=DIRTY_DIR):
    # Check if directories exist to provide a clearer error message
    if not os.path.isdir(clean_dir) or not os.path.isdir(dirty_dir):
        print(f"Error: One or both dataset directories not found.")
        print(f"  - Searched for clean directory: {os.path.abspath(clean_dir)}")
        print(f"  - Searched for dirty directory: {os.path.abspath(dirty_dir)}")
        print("Please ensure the `LAYOUT_NAME` is correct and the data generation pipeline has been run.")
        return

    clean_files = [f for f in os.listdir(clean_dir) if f.endswith('.npy')]
    dirty_files = [f for f in os.listdir(dirty_dir) if f.endswith('.npy')]

    if not clean_files or not dirty_files:
        print(f"Error: No .npy files found in '{clean_dir}' or '{dirty_dir}'.")
        return

    sample_clean = random.sample(clean_files, min(NUM_SAMPLES, len(clean_files)))
    sample_dirty = random.sample(dirty_files, min(NUM_SAMPLES, len(dirty_files)))

    # Setup the Matplotlib figure
    fig, axes = plt.subplots(2, NUM_SAMPLES, figsize=(18, 8))
    
    # Add super title with vital parameters AND Resolution
    fig.suptitle(f"Dataset Sanity Check | Window: {WINDOW_SIZE}nm | Clip/Image Size: {CLIP_SIZE}px | Res: {RESOLUTION:.1f}nm/px | Stride: {STRIDE}nm", 
                 fontsize=14, fontweight='bold', color='#333333')
    
    fig.canvas.manager.set_window_title('Dataset Sanity Check')
    fig.patch.set_facecolor('#f0f0f0') 

    print("="*50)
    print("INSPECTING RAW MATRICES (0 = Empty, 1 = Metal)")
    print("="*50)

    # Helper function to convert filename coordinates from nanometers to microns
    def format_coord_label(filename):
        try:
            # Expecting format "tile_X_Y.npy"
            parts = filename.replace('.npy', '').split('_')
            x_um = float(parts[1]) / 1000.0
            y_um = float(parts[2]) / 1000.0
            return f"X: {x_um}µm, Y: {y_um}µm"
        except Exception:
            return filename

    # Plot CLEAN samples (Top Row)
    for i, filename in enumerate(sample_clean):
        filepath = os.path.join(clean_dir, filename)
        matrix = np.load(filepath)
        
        # Convert the file name to micron format for the title
        coord_label = format_coord_label(filename)
        
        axes[0, i].imshow(matrix, cmap='gray_r')
        axes[0, i].set_title(f"CLEAN\n{coord_label}", color='green', fontweight='bold', fontsize=10)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        for spine in axes[0, i].spines.values():
            spine.set_edgecolor('black')

    # Plot VIOLATION samples (Bottom Row)
    for i, filename in enumerate(sample_dirty):
        filepath = os.path.join(dirty_dir, filename)
        matrix = np.load(filepath)
        
        # Convert the file name to micron format for the title
        coord_label = format_coord_label(filename)
        
        axes[1, i].imshow(matrix, cmap='gray_r')
        # FIX: Replaced "DIRTY" with standard industry terminology
        axes[1, i].set_title(f"Violation m1.2 : min. m1 spacing\n{coord_label}", color='red', fontweight='bold', fontsize=10)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        for spine in axes[1, i].spines.values():
            spine.set_edgecolor('black')

    print("\n" + "="*50)
    print("Opening visualizer window...")
    
    # h_pad prevents title overlap
    plt.tight_layout(h_pad=4.0)
    plt.subplots_adjust(top=0.88)
    plt.show()

if __name__ == "__main__":
    visualize_dataset_matrices()