import os
import sys
import time
import math
import numpy as np
import torch
import onnxruntime as ort
import multiprocessing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import (
    MODEL_WEIGHTS_PTH,
    MODEL_WEIGHTS_ONNX,
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
from GRAD_CAM import GradCAM, generate_gradcam

# ---------------------------------------------------------------------------
# CONFIGURATION — only edit this section before running
# ---------------------------------------------------------------------------
LAYOUT_NAME = "tt_um_cmos_inverter"

CONFIDENCE_THRESHOLD = 0.80

TILES_DIR = inference_dataset_dir(LAYOUT_NAME)
GRADCAM_OUT_DIR = gradcam_results_dir(LAYOUT_NAME)
REPORT_PATH = cnn_drc_report_path(LAYOUT_NAME)
INPUT_LAYOUT = layout_oas(LAYOUT_NAME)

BATCH_SIZE = 256
NMS_DISTANCE_THRESHOLD = 1600
# ---------------------------------------------------------------------------


def apply_nms(violations, distance_threshold):
    if not violations:
        return []

    violations = sorted(violations, key=lambda v: v[2], reverse=True)
    merged = []

    while violations:
        best = violations.pop(0)
        merged.append(best)
        keep = []
        for v in violations:
            dist = math.hypot(best[0] - v[0], best[1] - v[1])
            if dist > distance_threshold:
                keep.append(v)
        violations = keep

    return merged


def run_inference_with_gradcam(
    tiles_dir,
    pth_weights,
    onnx_weights,
    threshold,
    gradcam_dir,
    report_path,
    batch_size,
    input_layout_path=None,
    layout_name=None,
):
    if not os.path.exists(onnx_weights):
        print(f"[!] ONNX model not found at '{onnx_weights}'. Please run export_to_onnx.py first.")
        return

    os.makedirs(gradcam_dir, exist_ok=True)
    report_path = str(Path(report_path))

    tiles_path = os.path.join(tiles_dir, "tiles.npy")
    coords_path = os.path.join(tiles_dir, "coords.npy")

    if not os.path.exists(tiles_path) or not os.path.exists(coords_path):
        print(f"[!] Dataset files not found in {tiles_dir}.")
        return

    print(f"[*] Loading dataset into RAM from '{tiles_dir}'...")
    t0 = time.time()
    all_tiles_np = np.load(tiles_path)
    all_coords_np = np.load(coords_path)
    total_tiles = len(all_tiles_np)
    print(f"[*] Loaded {total_tiles} tiles in {time.time() - t0:.2f}s")

    cpu_cores = multiprocessing.cpu_count()
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = cpu_cores

    print(f"[*] Initializing ONNX C++ Engine with {cpu_cores} threads...")
    ort_session = ort.InferenceSession(onnx_weights, sess_options, providers=["CPUExecutionProvider"])
    input_name = ort_session.get_inputs()[0].name

    print("\n" + "=" * 55)
    print(" PHASE 1: Scanning all tiles for violations (ONNX)...")
    print("=" * 55)

    raw_violations = []
    phase1_start = time.time()

    for i in range(0, total_tiles, batch_size):
        batch_np = all_tiles_np[i : i + batch_size].astype(np.float32)
        batch_inputs = np.expand_dims(batch_np, axis=1)

        logits = ort_session.run(None, {input_name: batch_inputs})[0]

        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        for j, prob_arr in enumerate(probs):
            prob = prob_arr[1]
            if prob >= threshold:
                idx = i + j
                x, y = all_coords_np[idx]
                raw_violations.append((x, y, prob, idx))

        processed = min(i + batch_size, total_tiles)
        if processed % (batch_size * 10) == 0 or processed == total_tiles:
            pct = (processed / total_tiles) * 100
            elapsed = time.time() - phase1_start
            print(f"[>] {pct:.1f}% ({processed}/{total_tiles}) | Flagged: {len(raw_violations)} | Elapsed: {elapsed:.1f}s")

    phase1_time = time.time() - phase1_start
    print(f"\n[+] Phase 1 complete in {phase1_time:.1f}s — {len(raw_violations)} raw violation(s) found.")

    merged_violations = apply_nms(raw_violations, NMS_DISTANCE_THRESHOLD)
    if len(raw_violations) > 0:
        print(f"[*] NMS reduced {len(raw_violations)} overlapping detections into {len(merged_violations)} unique regions.")

    phase2_start = time.time()

    if merged_violations:
        print("\n" + "=" * 55)
        print(f" PHASE 2: Running Grad-CAM on {len(merged_violations)} unique violation(s)...")
        print("=" * 55)

        device = torch.device("cpu")
        model = NCSU_DRCNN().to(device)
        model.load_state_dict(torch.load(pth_weights, map_location=device))
        model.eval()

        grad_cam = GradCAM(model, model.conv4)

        for v_idx, (x, y, prob, orig_idx) in enumerate(merged_violations, 1):
            matrix = all_tiles_np[orig_idx].astype(np.float32)
            save_path = os.path.join(gradcam_dir, f"gradcam_x{x}_y{y}.png")

            generate_gradcam(matrix, grad_cam, device, save_path)
            print(f"[>] ({v_idx}/{len(merged_violations)}) tile_x{x}_y{y} | conf: {prob:.2%} -> {os.path.basename(save_path)}")

    phase2_time = time.time() - phase2_start
    total_time = phase1_time + phase2_time

    print("\n" + "=" * 55)
    if not merged_violations:
        print("[*] LAYOUT IS CLEAN — No DRC violations found.")
        write_cnn_drc_report(
            report_path,
            header_lines=[
                f"Raw hits      : {len(raw_violations)}",
                f"Unique (NMS)  : 0",
            ],
        )
        print(f"[*] Report saved to '{report_path}'")
    else:
        print(f"[!] FOUND {len(merged_violations)} POTENTIAL VIOLATION(S)!")

        violation_lines = []
        for x, y, prob, _ in merged_violations:
            line = f"Location: x{x}_y{y} | Confidence: {prob:.2%}"
            print(f"  - {line}")
            violation_lines.append(line)

        write_cnn_drc_report(
            report_path,
            violation_lines=violation_lines,
            header_lines=[
                f"Raw hits      : {len(raw_violations)}",
                f"Unique (NMS)  : {len(merged_violations)}",
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

    print(f"\n[*] Total Inference time: {total_time:.1f}s")
    print("=" * 55)


if __name__ == "__main__":
    run_inference_with_gradcam(
        str(TILES_DIR),
        str(MODEL_WEIGHTS_PTH),
        str(MODEL_WEIGHTS_ONNX),
        CONFIDENCE_THRESHOLD,
        str(GRADCAM_OUT_DIR),
        str(REPORT_PATH),
        BATCH_SIZE,
        input_layout_path=str(INPUT_LAYOUT),
        layout_name=LAYOUT_NAME,
    )
