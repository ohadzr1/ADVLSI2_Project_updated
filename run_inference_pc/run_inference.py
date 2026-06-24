import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import (
    MODEL_WEIGHTS_PTH,
    add_generate_training_dataset_scripts_to_syspath,
    inference_dataset_dir,
    gradcam_results_dir,
    cnn_drc_report_path,
    cnn_violation_mask_gds,
    layout_oas,
    merged_layout_with_cnn_mask,
)

add_generate_training_dataset_scripts_to_syspath()
from build_cnn_violation_mask_gds import build_mask_and_merge, write_cnn_drc_report

from define_cnn_model import NCSU_DRCNN
from GRAD_CAM import generate_gradcam

# ---------------------------------------------------------------------------
# CONFIGURATION  — only edit this section before running
# ---------------------------------------------------------------------------
LAYOUT_NAME = "tt_um_yen"
CONFIDENCE_THRESHOLD = 0.80

TILES_DIR = inference_dataset_dir(LAYOUT_NAME)
MODEL_WEIGHTS = MODEL_WEIGHTS_PTH
GRADCAM_OUTPUT_DIR = gradcam_results_dir(LAYOUT_NAME)
REPORT_PATH = cnn_drc_report_path(LAYOUT_NAME)
INPUT_LAYOUT = layout_oas(LAYOUT_NAME)
# ---------------------------------------------------------------------------


def _iter_tiles(tiles_dir):
    tiles_path = os.path.join(tiles_dir, "tiles.npy")
    coords_path = os.path.join(tiles_dir, "coords.npy")

    if os.path.exists(tiles_path) and os.path.exists(coords_path):
        tiles = np.load(tiles_path)
        coords = np.load(coords_path)
        for i in range(len(tiles)):
            x, y = int(coords[i][0]), int(coords[i][1])
            yield x, y, tiles[i]
        return

    npy_files = sorted(
        f for f in os.listdir(tiles_dir)
        if f.startswith("tile_x") and f.endswith(".npy")
    )
    for file_name in npy_files:
        stem = file_name.replace("tile_", "").replace(".npy", "")
        x_str, y_str = stem.split("_y", 1)
        x = int(x_str.replace("x", ""))
        y = int(y_str)
        yield x, y, np.load(os.path.join(tiles_dir, file_name))


def run_inference_with_gradcam(
    tiles_dir,
    model_weights,
    threshold,
    gradcam_dir,
    report_path,
    input_layout_path=None,
    layout_name=None,
):
    tiles_dir = str(Path(tiles_dir))
    model_weights = str(Path(model_weights))
    gradcam_dir = str(Path(gradcam_dir))
    report_path = str(Path(report_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    model = NCSU_DRCNN().to(device)
    if not os.path.exists(model_weights):
        print(f"[!] Model weights not found at '{model_weights}'. Aborting.")
        return
    model.load_state_dict(
        torch.load(model_weights, map_location=device, weights_only=True)
    )
    model.eval()
    print(f"[*] Model loaded from '{model_weights}'")

    if not os.path.isdir(tiles_dir):
        print(f"[!] Inference dataset directory not found at '{tiles_dir}'. Aborting.")
        return

    tile_iter = list(_iter_tiles(tiles_dir))
    if not tile_iter:
        print(f"[!] No tiles found in '{tiles_dir}'. Aborting.")
        return

    os.makedirs(gradcam_dir, exist_ok=True)

    violations = []
    start_time = time.time()
    total = len(tile_iter)

    print(f"[*] Scanning {total} tiles from '{tiles_dir}' "
          f"with confidence threshold {threshold:.0%}...")
    print(f"[*] Grad-CAM heatmaps will be saved to '{gradcam_dir}'")
    print("-" * 50)

    with torch.no_grad():
        for i, (x, y, matrix) in enumerate(tile_iter):
            matrix = matrix.astype(np.float32)
            tensor = torch.from_numpy(matrix).unsqueeze(0).unsqueeze(0).to(device)
            probs = torch.softmax(model(tensor), dim=1)
            violation_prob = probs[0][1].item()

            if violation_prob >= threshold:
                violations.append((x, y, violation_prob))
                gradcam_path = os.path.join(gradcam_dir, f"gradcam_x{x}_y{y}.png")
                generate_gradcam(matrix, model, device, save_path=gradcam_path)

            if (i + 1) % 500 == 0 or (i + 1) == total:
                elapsed = time.time() - start_time
                progress = (i + 1) / total * 100
                print(
                    f"[>] Progress: {progress:.1f}% ({i+1}/{total}) | "
                    f"Violations found: {len(violations)} | "
                    f"Elapsed: {elapsed:.1f}s"
                )

    elapsed_total = time.time() - start_time

    print("\n" + "=" * 50)
    if not violations:
        print("[*] LAYOUT IS CLEAN — No DRC violations found.")
        write_cnn_drc_report(
            report_path,
            header_lines=[
                f"Model weights : {model_weights}",
                f"Tiles dir     : {tiles_dir}",
                f"Threshold     : {threshold:.0%}",
                f"Total tiles   : {total}",
                f"Violations    : 0",
            ],
        )
        print(f"[*] Report saved to '{report_path}'")
    else:
        print(f"[!] FOUND {len(violations)} POTENTIAL VIOLATIONS!")
        print(f"[*] Grad-CAM heatmaps saved in '{gradcam_dir}'")
        print("=" * 50)

        violation_lines = []
        for x, y, prob in violations:
            line = f"Location: x{x}_y{y} | Confidence: {prob:.2%}"
            print(f"  - {line}")
            violation_lines.append(line)

        write_cnn_drc_report(
            report_path,
            violation_lines=violation_lines,
            header_lines=[
                f"Model weights : {model_weights}",
                f"Tiles dir     : {tiles_dir}",
                f"Threshold     : {threshold:.0%}",
                f"Total tiles   : {total}",
                f"Violations    : {len(violations)}",
            ],
        )
        print(f"\n[*] Full report saved to '{report_path}'")

        if input_layout_path and layout_name:
            build_mask_and_merge(
                input_layout_path=str(input_layout_path),
                report_path=report_path,
                mask_output_path=str(cnn_violation_mask_gds(layout_name)),
                merged_output_path=str(merged_layout_with_cnn_mask(layout_name)),
            )

    print(f"\n[*] Done! Total time: {elapsed_total:.1f}s")
    print("=" * 50)


if __name__ == "__main__":
    run_inference_with_gradcam(
        tiles_dir=TILES_DIR,
        model_weights=MODEL_WEIGHTS,
        threshold=CONFIDENCE_THRESHOLD,
        gradcam_dir=GRADCAM_OUTPUT_DIR,
        report_path=REPORT_PATH,
        input_layout_path=INPUT_LAYOUT,
        layout_name=LAYOUT_NAME,
    )
