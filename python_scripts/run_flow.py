import os
import shutil
import time
import argparse
import glob

# Import core functions from your scripts
from run_full_drc import run_full_drc
from extract_m1 import extract_m1
from extract_drc_mask_from_rdb import extract_drc_mask_from_rdb
from inject_drc_error import inject_isolated_m1_2_errors
from generate_training_dataset import generate_dataset
from visualize_dataset_matrices import visualize_dataset_matrices

# --- CONFIGURATION ---
DEFAULT_LAYOUT_NAME = "tt_um_cmos_inverter"
LAYOUTS_DIR = "real_layouts_tt"
ERROR_COUNT = 400
COMBINED_DATASET_DIR = "outputs/combined_training_dataset"


def build_paths(layout_name):
    base_out_dir        = f"outputs/{layout_name}"
    dataset_output      = f"{base_out_dir}/dataset_output"
    training_dataset    = f"{base_out_dir}/training_dataset"
    original_gds        = f"{LAYOUTS_DIR}/{layout_name}.oas"
    extracted_m1_gds    = f"{dataset_output}/{layout_name}_M1.gds"
    injected_gds        = f"{dataset_output}/{layout_name}_M1_m1_2_Marked.gds"
    drc_report          = f"{dataset_output}/sky130_drc.txt"
    mask_file           = f"{dataset_output}/drc_mask_layer_255.oas"
    return dict(
        base_out_dir=base_out_dir,
        dataset_output=dataset_output,
        training_dataset=training_dataset,
        original_gds=original_gds,
        extracted_m1_gds=extracted_m1_gds,
        injected_gds=injected_gds,
        drc_report=drc_report,
        mask_file=mask_file,
    )


def clean_layout_folder(paths):
    """Delete old data to prevent dataset contamination, then recreate folders."""
    if os.path.exists(paths["base_out_dir"]):
        print(f"[*] Cleaning up old '{paths['base_out_dir']}' directory...")
        shutil.rmtree(paths["base_out_dir"])
    os.makedirs(paths["dataset_output"], exist_ok=True)
    os.makedirs(paths["training_dataset"], exist_ok=True)
    time.sleep(0.5)


def run_single_layout(layout_name, tile_prefix=""):
    """Run the full DRC pipeline for one layout. Returns (v_count, c_count) or None on failure."""
    paths = build_paths(layout_name)

    print("=" * 60)
    print(f" STARTING FULL AUTONOMOUS DRC PIPELINE FOR: {layout_name}")
    print("=" * 60)

    clean_layout_folder(paths)

    print("\n>>> STEP 1: Extracting M1 (Layer 68/20) from original layout <<<")
    extract_m1(paths["original_gds"], paths["dataset_output"])

    print("\n>>> STEP 2: Injecting NEW DRC Errors into the M1 layout <<<")
    inject_isolated_m1_2_errors(paths["extracted_m1_gds"], num_errors=ERROR_COUNT)

    print(f"\n>>> STEP 3: Running KLayout DRC Engine on {paths['injected_gds']} <<<")
    drc_success = run_full_drc(paths["injected_gds"], paths["drc_report"])
    if not drc_success:
        print(f"Pipeline aborted for {layout_name} due to DRC failure.")
        return None

    print("\n>>> STEP 4: Extracting verified violations from RDB to Mask <<<")
    total_drc_errors_found = extract_drc_mask_from_rdb(paths["drc_report"], paths["mask_file"])

    print("\n>>> STEP 5: Generating Dataset Tiles (Matrix conversion) <<<")
    v_count, c_count = generate_dataset(
        paths["injected_gds"],
        paths["training_dataset"],
        tile_prefix=tile_prefix,
    )

    print("\n>>> STEP 6: Visualizing Results (Sanity Check) <<<")
    visualize_dataset_matrices(
        f"{paths['training_dataset']}/clean",
        f"{paths['training_dataset']}/dirty",
    )

    print("\n" + "=" * 60)
    print(f" PIPELINE COMPLETED: {layout_name}")
    print(f"   * Layout Errors Injected:    {ERROR_COUNT}")
    print(f"   * Layout Errors Found (DRC): {total_drc_errors_found}")
    print(f"   * Violation Images (Dirty):  {v_count}")
    print(f"   * Clean Images Generated:    {c_count}")
    print(f"   * Total Training Samples:    {v_count + c_count}")
    print("=" * 60)

    return v_count, c_count, paths["training_dataset"]


def run_all_layouts():
    """Run the pipeline for every .oas layout in LAYOUTS_DIR and produce one combined zip."""
    oas_files = sorted(glob.glob(f"{LAYOUTS_DIR}/*.oas"))
    if not oas_files:
        print(f"[!] No .oas files found in '{LAYOUTS_DIR}'. Aborting.")
        return

    layout_names = [os.path.splitext(os.path.basename(f))[0] for f in oas_files]
    print(f"[*] Found {len(layout_names)} layouts: {', '.join(layout_names)}")

    # Prepare the combined output folder
    if os.path.exists(COMBINED_DATASET_DIR):
        print(f"[*] Cleaning up old combined dataset directory...")
        shutil.rmtree(COMBINED_DATASET_DIR)
    os.makedirs(f"{COMBINED_DATASET_DIR}/clean", exist_ok=True)
    os.makedirs(f"{COMBINED_DATASET_DIR}/dirty", exist_ok=True)

    grand_v = 0
    grand_c = 0
    failed = []

    for layout_name in layout_names:
        tile_prefix = f"{layout_name}_"
        result = run_single_layout(layout_name, tile_prefix=tile_prefix)
        if result is None:
            failed.append(layout_name)
            continue

        v_count, c_count, training_dir = result
        grand_v += v_count
        grand_c += c_count

        # Copy tiles (already prefixed) into the combined folder
        for split in ("clean", "dirty"):
            src_dir = os.path.join(training_dir, split)
            dst_dir = os.path.join(COMBINED_DATASET_DIR, split)
            for npy_file in glob.glob(os.path.join(src_dir, "*.npy")):
                shutil.copy2(npy_file, dst_dir)

    # Create a single combined zip archive
    zip_path = shutil.make_archive(COMBINED_DATASET_DIR, "zip", root_dir=COMBINED_DATASET_DIR)

    print("\n" + "=" * 60)
    print(" ALL LAYOUTS PROCESSED — COMBINED DATASET SUMMARY")
    print("=" * 60)
    print(f"   * Layouts processed:         {len(layout_names) - len(failed)}/{len(layout_names)}")
    if failed:
        print(f"   * Failed layouts:            {', '.join(failed)}")
    print(f"   * Total Violation (Dirty):   {grand_v}")
    print(f"   * Total Clean:               {grand_c}")
    print(f"   * Total Training Samples:    {grand_v + grand_c}")
    print(f"   * Combined archive:          {zip_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run the full DRC training-data pipeline."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--layout",
        metavar="NAME",
        help=f"Name of a single layout in {LAYOUTS_DIR}/ (without extension). "
             f"Defaults to '{DEFAULT_LAYOUT_NAME}'.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help=f"Run the pipeline on every .oas file in {LAYOUTS_DIR}/ and "
             "produce one combined training dataset zip.",
    )
    args = parser.parse_args()

    if args.all:
        run_all_layouts()
    else:
        layout_name = args.layout if args.layout else DEFAULT_LAYOUT_NAME
        run_single_layout(layout_name)


if __name__ == "__main__":
    main()
