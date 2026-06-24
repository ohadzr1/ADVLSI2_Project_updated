import os
import sys
import time
import tempfile
import numpy as np
import klayout.db as db
from PIL import Image, ImageDraw
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import add_generate_training_dataset_scripts_to_syspath, layout_oas, inference_dataset_dir

add_generate_training_dataset_scripts_to_syspath()
from extract_m1 import extract_m1

# --- CONFIGURATION ---
LAYOUT_NAME = "tt_um_cmos_inverter"
INPUT_GDS = layout_oas(LAYOUT_NAME)
OUTPUT_DIR = inference_dataset_dir(LAYOUT_NAME)
METAL_LAYER = (68, 20)
PHYSICAL_SIZE = 1600
STRIDE = 1500
IMAGE_SIZE = 200
SCALE_FACTOR = IMAGE_SIZE / PHYSICAL_SIZE


def generate_inference_dataset(
    input_gds=INPUT_GDS,
    output_dir=OUTPUT_DIR,
    batch_queue=None,
    batch_size=256,
):
    input_gds = str(Path(input_gds))
    output_dir = str(Path(output_dir))
    start_time = time.time()

    if batch_queue is None:
        os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as generated_dir:
        print(f"[*] Extracting Metal 1 from {input_gds}...")
        extract_m1(input_gds, generated_dir)
        file_base = os.path.splitext(os.path.basename(input_gds))[0]
        m1_gds = os.path.join(generated_dir, f"{file_base}_M1.gds")

        print(f"[*] Loading layout {m1_gds}...")
        layout = db.Layout()
        layout.read(m1_gds)
        top_cell = layout.top_cell()
        layer_idx = layout.find_layer(*METAL_LAYER)
        metal_region = db.Region(top_cell.begin_shapes_rec(layer_idx))
        bbox = metal_region.bbox()

        x_range = range(bbox.left, bbox.right - PHYSICAL_SIZE, STRIDE)
        y_range = range(bbox.bottom, bbox.top - PHYSICAL_SIZE, STRIDE)
        total_steps = len(x_range) * len(y_range)

        print(f"[*] Layout Size: {bbox.width()/1000:.2f}um x {bbox.height()/1000:.2f}um")
        print(f"[*] Strategy: Window={PHYSICAL_SIZE}nm, Stride={STRIDE}nm")
        print(f"[*] Total windows to process: {total_steps}")

        processed_count = 0
        current_batch_matrices = []
        current_batch_coords = []
        all_tiles_list = []
        all_coords_list = []

        for x in x_range:
            for y in y_range:
                processed_count += 1

                tile_box = db.Box(x, y, x + PHYSICAL_SIZE, y + PHYSICAL_SIZE)
                tile_metal = metal_region & db.Region(tile_box)

                if not tile_metal.is_empty():
                    img = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
                    draw = ImageDraw.Draw(img)

                    for poly in tile_metal.each():
                        shifted_poly = poly.moved(-x, -y)
                        pts = [
                            (pt.x * SCALE_FACTOR, pt.y * SCALE_FACTOR)
                            for pt in shifted_poly.each_point_hull()
                        ]
                        if len(pts) >= 3:
                            draw.polygon(pts, fill=1)

                    matrix = np.array(img, dtype=np.uint8)
                    matrix = np.flipud(matrix)

                    if np.mean(matrix) > 0.01:
                        all_tiles_list.append(matrix)
                        all_coords_list.append((x, y))
                        if batch_queue is not None:
                            current_batch_matrices.append(matrix)
                            current_batch_coords.append((x, y))

                            if len(current_batch_matrices) == batch_size:
                                batch_queue.put(
                                    (np.stack(current_batch_matrices), np.array(current_batch_coords))
                                )
                                current_batch_matrices = []
                                current_batch_coords = []

                if processed_count % 500 == 0 or processed_count == total_steps:
                    progress = (processed_count / total_steps) * 100
                    if batch_queue is None:
                        print(f"[>] Progress: {progress:.1f}% | Tiles collected: {len(all_tiles_list)}")

        if batch_queue is None:
            saved_count = len(all_tiles_list)
            if saved_count > 0:
                tiles_array = np.stack(all_tiles_list).astype(np.uint8)
                coords_array = np.array(all_coords_list, dtype=np.int32)

                np.save(os.path.join(output_dir, "tiles.npy"), tiles_array)
                np.save(os.path.join(output_dir, "coords.npy"), coords_array)

                print(
                    f"[*] Saved tiles.npy  — shape {tiles_array.shape}, "
                    f"{tiles_array.nbytes / 1e6:.1f} MB"
                )
                print(f"[*] Saved coords.npy — shape {coords_array.shape}")

            print(
                f"\n Done! Processed {total_steps} windows. "
                f"Saved {saved_count} tiles to '{output_dir}'."
            )
            print(f"[*] Total run time: {time.time() - start_time:.2f} seconds")
        else:
            if len(current_batch_matrices) > 0:
                batch_queue.put(
                    (np.stack(current_batch_matrices), np.array(current_batch_coords))
                )

            saved_count = len(all_tiles_list)
            if saved_count > 0:
                os.makedirs(output_dir, exist_ok=True)
                tiles_array = np.stack(all_tiles_list).astype(np.uint8)
                coords_array = np.array(all_coords_list, dtype=np.int32)
                np.save(os.path.join(output_dir, "tiles.npy"), tiles_array)
                np.save(os.path.join(output_dir, "coords.npy"), coords_array)
                print(
                    f"[*] Saved tiles.npy  — shape {tiles_array.shape}, "
                    f"{tiles_array.nbytes / 1e6:.1f} MB"
                )
                print(f"[*] Saved coords.npy — shape {coords_array.shape}")

            batch_queue.put(None)
            print("[*] Finished generating tiles.")


if __name__ == "__main__":
    generate_inference_dataset()
