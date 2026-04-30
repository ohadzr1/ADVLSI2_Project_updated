import os
import shutil
import time

# Import core functions from your scripts
from run_full_drc import run_full_drc
from extract_m1 import extract_m1
from extract_drc_mask_from_rdb import extract_drc_mask_from_rdb
from inject_drc_error import inject_isolated_m1_2_errors
from generate_training_dataset import generate_dataset
from visualize_dataset_matrices import visualize_dataset_matrices

# --- CONFIGURATION ---
LAYOUT_NAME = "tt_um_yen"
ORIGINAL_GDS = f"real_layouts_tt/{LAYOUT_NAME}.oas" 
BASE_OUT_DIR = f"outputs/{LAYOUT_NAME}"
DATASET_OUTPUT = f"{BASE_OUT_DIR}/dataset_output"
TRAINING_DATASET = f"{BASE_OUT_DIR}/training_dataset"

EXTRACTED_M1_GDS = f"{DATASET_OUTPUT}/{LAYOUT_NAME}_M1.gds"
INJECTED_GDS = f"{DATASET_OUTPUT}/{LAYOUT_NAME}_M1_m1_2_Marked.gds" 
DRC_REPORT = f"{DATASET_OUTPUT}/sky130_drc.txt"
MASK_FILE = f"{DATASET_OUTPUT}/drc_mask_layer_255.oas"
ERROR_COUNT = 400              

def clean_dataset_folder():
    """Automatically delete old data to prevent dataset contamination."""
    if os.path.exists(BASE_OUT_DIR):
        print(f"[*] Cleaning up old '{BASE_OUT_DIR}' directory...")
        shutil.rmtree(BASE_OUT_DIR)

            
    # Create the required folders for the new layout
    os.makedirs(DATASET_OUTPUT, exist_ok=True)
    os.makedirs(TRAINING_DATASET, exist_ok=True)
    time.sleep(0.5)

def main():
    print("="*60)
    print(f" STARTING FULL AUTONOMOUS DRC PIPELINE FOR: {LAYOUT_NAME}")  
    print("="*60)

    # Step 0: Cleanup
    clean_dataset_folder()

    # Step 1: Extract Metal
    print("\n>>> STEP 1: Extracting M1 (Layer 68/20) from original layout <<<")
    extract_m1(ORIGINAL_GDS, DATASET_OUTPUT)

    # Step 2: Inject Errors
    print("\n>>> STEP 2: Injecting NEW DRC Errors into the M1 layout <<<")
    inject_isolated_m1_2_errors(EXTRACTED_M1_GDS, num_errors=ERROR_COUNT)
    
    # Step 3: Run DRC
    print(f"\n>>> STEP 3: Running KLayout DRC Engine on {INJECTED_GDS} <<<")
    drc_success = run_full_drc(INJECTED_GDS, DRC_REPORT)
    if not drc_success:
        print("Pipeline aborted due to DRC failure.")
        return

    # Step 4: Process RDB
    print("\n>>> STEP 4: Extracting verified violations from RDB to Mask <<<")
    total_drc_errors_found = extract_drc_mask_from_rdb(DRC_REPORT, MASK_FILE)
    
    # Step 5: Tiling & Matrix Conversion
    print("\n>>> STEP 5: Generating Dataset Tiles (Matrix conversion) <<<")
    # Capture the image counts returned from the function
    v_count, c_count = generate_dataset(INJECTED_GDS, TRAINING_DATASET)
    
    # Step 6: Visualization
    print("\n>>> STEP 6: Visualizing Results (Sanity Check) <<<")
    visualize_dataset_matrices(f"{TRAINING_DATASET}/clean", f"{TRAINING_DATASET}/dirty")
    
    # --- FINAL SUMMARY REPORT ---
    print("\n" + "="*60)
    print(" FULL PIPELINE COMPLETED SUCCESSFULLY!")
    print(f" DATASET SUMMARY:")
    print(f"   * Layout Errors Injected:    {ERROR_COUNT}")
    print(f"   * Layout Errors Found (DRC): {total_drc_errors_found}")
    print(f"   --------------------------------------")
    print(f"   * Violation Images (Dirty):  {v_count}")
    print(f"   * Clean Images Generated:    {c_count}")
    print(f"   * Total Training Samples:    {v_count + c_count}")
    print("="*60)

if __name__ == "__main__":
    main()