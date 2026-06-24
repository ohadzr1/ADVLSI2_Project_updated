import argparse
import os
import re
import sys
import klayout.db as db
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import (
    layout_oas,
    cnn_drc_report_path,
    cnn_violation_mask_gds,
    merged_layout_with_cnn_mask,
)

# Must match the inference tiling window size from generate_inference_dataset.py
PHYSICAL_SIZE = 1600
LAYOUT_NAME = "tt_um_yen_1err"

DEFAULT_INPUT_LAYOUT = layout_oas(LAYOUT_NAME)
DEFAULT_REPORT_PATH = cnn_drc_report_path(LAYOUT_NAME)
DEFAULT_MASK_OUTPUT = cnn_violation_mask_gds(LAYOUT_NAME)
DEFAULT_MERGED_OUTPUT = merged_layout_with_cnn_mask(LAYOUT_NAME)

# A dedicated layer/datatype for CNN mask polygons
# (in KLayout layer properties you can color this layer red for visualization)
MASK_LAYER = (81, 63)

LOCATION_RE = re.compile(
    r"Location:\s*x(?P<x>-?\d+)_y(?P<y>-?\d+)\s*\|\s*Confidence:\s*(?P<conf>[0-9]*\.?[0-9]+)%"
)

CLEAN_LAYOUT_MESSAGE = "Layout is clean - no m1.2 errors found."


def write_cnn_drc_report(report_path, violation_lines=None, header_lines=None):
    """Write CNN DRC report to disk. Empty violation_lines writes a clean-layout message."""
    report_path = str(Path(report_path))
    parent = os.path.dirname(report_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("CNN DRC Violations Report\n")
        f.write("=========================\n")
        if header_lines:
            for line in header_lines:
                f.write(f"{line}\n")
        if not violation_lines:
            if header_lines:
                f.write("\n")
            f.write(f"{CLEAN_LAYOUT_MESSAGE}\n")
        else:
            if header_lines:
                f.write("\n")
            for line in violation_lines:
                f.write(f"{line}\n")


def parse_cnn_report(report_path, min_confidence=0.0):
    violations = []
    with open(report_path, "r", encoding="utf-8") as f:
        for line in f:
            m = LOCATION_RE.search(line)
            if not m:
                continue
            x = int(m.group("x"))
            y = int(m.group("y"))
            conf = float(m.group("conf"))
            if conf >= min_confidence:
                violations.append((x, y, conf))
    return violations


def build_mask_and_merge(
    input_layout_path,
    report_path,
    mask_output_path,
    merged_output_path,
    tile_size=PHYSICAL_SIZE,
    mask_layer=MASK_LAYER,
    min_confidence=0.0,
):
    if not os.path.exists(input_layout_path):
        raise FileNotFoundError(f"Input layout not found: {input_layout_path}")
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"CNN report not found: {report_path}")

    violations = parse_cnn_report(report_path, min_confidence=min_confidence)
    if not violations:
        print("[!] No violations matched the report/threshold. Creating empty mask/merged outputs.")

    # Load original layout and append violation boxes on a dedicated mask layer.
    merged_layout = db.Layout()
    merged_layout.read(input_layout_path)
    top_cell = merged_layout.top_cell()
    mask_layer_idx = merged_layout.layer(*mask_layer)

    for x, y, _ in violations:
        top_cell.shapes(mask_layer_idx).insert(db.Box(x, y, x + tile_size, y + tile_size))

    os.makedirs(os.path.dirname(merged_output_path) or ".", exist_ok=True)
    merged_layout.write(merged_output_path)

    # Create standalone mask layout (same DBU for proper overlay/alignment).
    mask_layout = db.Layout()
    mask_layout.dbu = merged_layout.dbu
    mask_top = mask_layout.create_cell(top_cell.name)
    standalone_mask_layer_idx = mask_layout.layer(*mask_layer)

    for x, y, _ in violations:
        mask_top.shapes(standalone_mask_layer_idx).insert(db.Box(x, y, x + tile_size, y + tile_size))

    os.makedirs(os.path.dirname(mask_output_path) or ".", exist_ok=True)
    mask_layout.write(mask_output_path)

    print(f"[*] Violations parsed: {len(violations)}")
    print(f"[*] Tile size used: {tile_size} nm")
    print(f"[*] Mask layer: {mask_layer[0]}/{mask_layer[1]}")
    print(f"[*] Standalone mask written to: {mask_output_path}")
    print(f"[*] Merged layout written to: {merged_output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a GDS violation mask from CNN report lines such as "
            "'Location: x101500_y22500 | Confidence: 85.69%' "
            "and merge it with the original layout."
        )
    )
    parser.add_argument(
        "--input-layout",
        default=str(DEFAULT_INPUT_LAYOUT),
        help=f"Path to original layout (.gds/.oas) (default: {DEFAULT_INPUT_LAYOUT})",
    )
    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT_PATH),
        help=f"Path to CNN violation report text file (default: {DEFAULT_REPORT_PATH})",
    )
    parser.add_argument(
        "--mask-output",
        default=str(DEFAULT_MASK_OUTPUT),
        help=f"Output path for standalone mask GDS (default: {DEFAULT_MASK_OUTPUT})",
    )
    parser.add_argument(
        "--merged-output",
        default=str(DEFAULT_MERGED_OUTPUT),
        help=f"Output path for merged layout GDS (default: {DEFAULT_MERGED_OUTPUT})",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=PHYSICAL_SIZE,
        help=f"Violation box size in layout units (default: {PHYSICAL_SIZE})",
    )
    parser.add_argument(
        "--mask-layer",
        type=str,
        default=f"{MASK_LAYER[0]}/{MASK_LAYER[1]}",
        help=f"Mask layer/datatype as LAYER/DATATYPE (default: {MASK_LAYER[0]}/{MASK_LAYER[1]})",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Only include violations with confidence >= this threshold (percent).",
    )
    args = parser.parse_args()

    if "/" not in args.mask_layer:
        raise ValueError("mask-layer must be in the format LAYER/DATATYPE (e.g. 255/0)")
    layer_s, dtype_s = args.mask_layer.split("/", 1)
    mask_layer = (int(layer_s), int(dtype_s))

    build_mask_and_merge(
        input_layout_path=args.input_layout,
        report_path=args.report,
        mask_output_path=args.mask_output,
        merged_output_path=args.merged_output,
        tile_size=args.tile_size,
        mask_layer=mask_layer,
        min_confidence=args.min_confidence,
    )


if __name__ == "__main__":
    main()
