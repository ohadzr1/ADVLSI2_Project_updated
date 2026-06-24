"""
Portable path helpers for this project.

All paths are resolved relative to the repository root (the directory
containing this file), so scripts work regardless of the current working
directory or machine.

Override KLayout location with the KLAYOUT_CMD environment variable.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

LAYOUTS_DIR = PROJECT_ROOT / "real_layouts_tt"
TRAINING_DATASETS_DIR = PROJECT_ROOT / "training_datasets"
INFERENCE_RESULTS_DIR = PROJECT_ROOT / "inference_results"
COMBINED_DATASET_DIR = TRAINING_DATASETS_DIR / "combined_training_dataset"
SKY130_DRC_SCRIPT = PROJECT_ROOT / "sky130_drc_deck" / "run_drc_full.lydrc"
MODEL_WEIGHTS_PTH = PROJECT_ROOT / "ncsu_drcnn_weights.pth"
MODEL_WEIGHTS_ONNX = PROJECT_ROOT / "ncsu_drcnn.onnx"
GENERATE_TRAINING_DATASET_SCRIPTS_DIR = PROJECT_ROOT / "generate_training_dataset_scripts"
RUN_INFERENCE_PC_OPTIMIZED_DIR = PROJECT_ROOT / "run_inference_pc_optimized"
RUN_INFERENCE_PC_DIR = PROJECT_ROOT / "run_inference_pc"


def add_project_root_to_syspath() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def add_generate_training_dataset_scripts_to_syspath() -> None:
    add_project_root_to_syspath()
    scripts = str(GENERATE_TRAINING_DATASET_SCRIPTS_DIR)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)


def layout_oas(name: str) -> Path:
    return LAYOUTS_DIR / f"{name}.oas"


def layout_output_dir(name: str) -> Path:
    """Per-layout directory under training_datasets (training pipeline)."""
    return TRAINING_DATASETS_DIR / name


def inference_layout_dir(name: str) -> Path:
    """Per-layout directory under inference_results (inference pipeline)."""
    return INFERENCE_RESULTS_DIR / name


def dataset_output_dir(name: str) -> Path:
    """Intermediate M1 extraction output for the training pipeline."""
    return layout_output_dir(name) / "dataset_output"


def training_dataset_dir(name: str) -> Path:
    return layout_output_dir(name) / "training_dataset"


def inference_dataset_dir(name: str) -> Path:
    return inference_layout_dir(name) / "inference_dataset"


def gradcam_results_dir(name: str) -> Path:
    return inference_layout_dir(name) / "gradcam_results"


def cnn_drc_report_path(name: str) -> Path:
    """CNN inference violation report (layout root, not under gradcam_results/)."""
    return inference_layout_dir(name) / "drc_report.txt"


def cnn_violation_mask_gds(name: str) -> Path:
    return inference_layout_dir(name) / "cnn_violation_mask.gds"


def merged_layout_with_cnn_mask(name: str) -> Path:
    return inference_layout_dir(name) / f"{name}_with_cnn_mask.gds"


def extracted_m1_gds(layout_name: str) -> Path:
    return dataset_output_dir(layout_name) / f"{layout_name}_M1.gds"


def injected_m1_gds(layout_name: str) -> Path:
    return dataset_output_dir(layout_name) / f"{layout_name}_M1_m1_2_Marked.gds"


def drc_report_path(layout_name: str) -> Path:
    return dataset_output_dir(layout_name) / "sky130_drc.txt"


def drc_mask_path(layout_name: str) -> Path:
    return dataset_output_dir(layout_name) / "drc_mask_layer_255.oas"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_klayout_executable() -> str:
    """
    Resolve the KLayout executable in this order:
      1. KLAYOUT_CMD or KLAYOUT environment variable
      2. `klayout` on PATH
      3. Common install locations (Windows / macOS / Linux)
      4. Fallback to `klayout` and let the shell resolve it
    """
    for env_name in ("KLAYOUT_CMD", "KLAYOUT"):
        env_value = os.environ.get(env_name)
        if env_value:
            candidate = Path(env_value)
            if candidate.exists():
                return str(candidate)
            if shutil.which(env_value):
                return shutil.which(env_value)

    on_path = shutil.which("klayout")
    if on_path:
        return on_path

    local_app_data = os.environ.get("LOCALAPPDATA", "")
    roaming_app_data = os.environ.get("APPDATA", "")
    candidates = [
        Path(roaming_app_data) / "KLayout" / "klayout_app.exe" if roaming_app_data else None,
        Path(local_app_data) / "KLayout" / "klayout_app.exe" if local_app_data else None,
        Path(roaming_app_data) / "KLayout" / "klayout.exe" if roaming_app_data else None,
        Path("C:/Program Files/KLayout/klayout.exe"),
        Path("/Applications/klayout.app/Contents/MacOS/klayout"),
        Path("/usr/bin/klayout"),
        Path("/usr/local/bin/klayout"),
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return str(candidate)

    return "klayout"
