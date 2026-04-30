import os
import numpy as np
import matplotlib.pyplot as plt
import random

# --- CONFIGURATION ---
# Path to the folder containing the generated .npy inference tiles
LAYOUT_NAME = "tt_um_yen"
DATASET_PATH = f"outputs/{LAYOUT_NAME}/inference_dataset"

# Add specific coordinates here to inspect them (e.g., "x7000_y7500")
#SPECIFIC_LOCATIONS = ["x151000_y58500","x151000_y147000","x16000_y48000","x20500_y129000","x67000_y100500","x67000_y46500"]
SPECIFIC_LOCATIONS = []

# Number of random tiles to show if SPECIFIC_LOCATIONS is empty
NUM_RANDOM_SAMPLES = 12

# Physical parameters corresponding to the Inference Generator
WINDOW_SIZE = 1600
CLIP_SIZE = 200
STRIDE = 1500
RESOLUTION = WINDOW_SIZE / CLIP_SIZE

def visualize_inference_dataset(folder_path, coords_list=None, num_random=12):
    # Load all available .npy files in the directory
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    
    if not all_files:
        print(f"[!] No .npy files found in {folder_path}")
        return

    files_to_extract = []
    
    # Selection Logic: Prioritize specific coordinates, otherwise sample randomly
    if coords_list:
        print(f"[*] Searching for {len(coords_list)} specific coordinates...")
        for coord in coords_list:
            matched = [f for f in all_files if f"tile_{coord}.npy" in f]
            if matched:
                files_to_extract.append(matched[0])
            else:
                print(f"[!] Warning: Could not find coordinate {coord} in the folder.")
    else:
        print(f"[*] No specific coordinates provided. Picking {num_random} random samples...")
        files_to_extract = random.sample(all_files, min(num_random, len(all_files)))

    if not files_to_extract:
        print("[!] No files to display.")
        return

    num_samples = len(files_to_extract)
    cols = min(6, num_samples)
    rows = (num_samples // cols) + (num_samples % cols > 0)
    if rows == 0: rows = 1
    
    # Initialize the Matplotlib figure grid
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 4 * rows))
    if num_samples == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()
    
    fig.suptitle(f"Inference Aligned View | Window: {WINDOW_SIZE}nm | Res: {RESOLUTION:.1f}nm/px", 
                 fontsize=14, fontweight='bold', color='#333333')
    fig.patch.set_facecolor('#f0f0f0') 

    # Helper function to convert filename coordinates from nm to microns
    def format_coord_label(filename):
        try:
            clean_str = filename.replace('.npy', '').replace('tile_', '')
            if '_y' in clean_str:
                parts = clean_str.replace('x', '').split('_y')
                x_um = float(parts[0]) / 1000.0
                y_um = float(parts[1]) / 1000.0
                return f"X: {x_um}um, Y: {y_um}um"
            return clean_str
        except Exception:
            return filename

    for i, file_name in enumerate(files_to_extract):
        file_path = os.path.join(folder_path, file_name)
        matrix = np.load(file_path)
        
        # --- ALIGNMENT LOGIC ---
        # The generator already applied np.flipud() before saving the file.
        # Displaying the matrix directly matches the Training visualization 
        # and correctly aligns with the KLayout Top-Down view.
        
        coord_label = format_coord_label(file_name)
        
        # Display using gray_r colormap: 1 (Metal) = Black, 0 (Empty) = White
        axes[i].imshow(matrix, cmap='gray_r')
        
        # Color coding: Red for specific targets, Blue for random unclassified tiles
        title_color = '#AA0000' if coords_list else '#0055AA'
        axes[i].set_title(f"INSPECTION TARGET\n{coord_label}", color=title_color, fontweight='bold', fontsize=10)
        
        # Remove axes ticks for cleaner visualization
        axes[i].set_xticks([]); axes[i].set_yticks([])
        for spine in axes[i].spines.values():
            spine.set_edgecolor('black')

    # Hide any unused subplot slots in the grid
    for j in range(num_samples, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    print(f"[*] Successfully displayed {num_samples} tiles.")

if __name__ == "__main__":
    visualize_inference_dataset(DATASET_PATH, coords_list=SPECIFIC_LOCATIONS)