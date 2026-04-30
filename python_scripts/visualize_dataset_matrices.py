import numpy as np
import matplotlib.pyplot as plt
import os
import random

# --- CONFIGURATION ---
CLEAN_DIR = "training_dataset/clean"
DIRTY_DIR = "training_dataset/dirty"
NUM_SAMPLES = 6 # 6 per row = 12 samples total

# Plot parameters corresponding to dataset generator
WINDOW_SIZE = 1600
CLIP_SIZE = 200
STRIDE = 400

# Calculate resolution automatically
RESOLUTION = WINDOW_SIZE / CLIP_SIZE

def visualize_dataset_matrices():
    clean_files = [f for f in os.listdir(CLEAN_DIR) if f.endswith('.npy')]
    dirty_files = [f for f in os.listdir(DIRTY_DIR) if f.endswith('.npy')]

    if not clean_files or not dirty_files:
        print("Error: Could not find .npy files in the dataset folders.")
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
        filepath = os.path.join(CLEAN_DIR, filename)
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
        filepath = os.path.join(DIRTY_DIR, filename)
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