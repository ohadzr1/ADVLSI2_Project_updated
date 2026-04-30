import klayout.db as db
import numpy as np
from PIL import Image, ImageDraw
import os

INPUT_LAYOUT = "tt_um_yen_M1_m1_2_Marked.gds" 
METAL_LAYER = (68, 20)

MASK_LAYOUT = INPUT_LAYOUT 
MASK_LAYER = (255, 0)

# --- PHYSICAL & IMAGE RESOLUTION SETTINGS ---
PHYSICAL_SIZE = 1600    
STRIDE = 400           
IMAGE_SIZE = 200        
MARGIN = 600
OUTPUT_DIR = "training_dataset"

def generate_dataset():
    layout_metal = db.Layout()
    layout_metal.read(INPUT_LAYOUT)
    top_cell_metal = layout_metal.top_cell()
    
    metal_idx = layout_metal.find_layer(*METAL_LAYER)
    metal_region = db.Region(top_cell_metal.begin_shapes_rec(metal_idx))

    mask_idx = layout_metal.find_layer(*MASK_LAYER)
    mask_region = db.Region(top_cell_metal.begin_shapes_rec(mask_idx))

    bbox = metal_region.bbox()

    os.makedirs(f"{OUTPUT_DIR}/clean", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/dirty", exist_ok=True)

    dirty_count = 0
    clean_count = 0
    discard_count = 0 
    scale_factor = IMAGE_SIZE / PHYSICAL_SIZE 

    print(f"Starting Tiling... Window: {PHYSICAL_SIZE}nm, Res: {PHYSICAL_SIZE/IMAGE_SIZE}nm/px")

    x_range = range(bbox.left, bbox.right - PHYSICAL_SIZE, STRIDE)
    y_range = range(bbox.bottom, bbox.top - PHYSICAL_SIZE, STRIDE)
    total_steps = len(x_range) * len(y_range)
    processed_count = 0

    # Main tiling loop: Iterate over the layout grid to extract physical windows and process them into image matrices.
    for x in x_range:
        for y in y_range:
            processed_count += 1
            # Print progress update every 10000 windows, or at the end, to improve user experience
            if processed_count % 10000 == 0 or processed_count == total_steps:
                # Calculate the percentage completed, with a check to prevent division by zero
                progress = (processed_count / total_steps) * 100 if total_steps > 0 else 100
                print(f"[>] Progress: {progress:.1f}% | Dirty: {dirty_count} | Clean: {clean_count}")
            
            tile_box = db.Box(x, y, x + PHYSICAL_SIZE, y + PHYSICAL_SIZE)
            tile_region = db.Region(tile_box)
            tile_mask_full = mask_region & tile_region
            has_error_anywhere = not tile_mask_full.is_empty()

            safe_box = db.Box(x + MARGIN, y + MARGIN, x + PHYSICAL_SIZE - MARGIN, y + PHYSICAL_SIZE - MARGIN)
            safe_region = db.Region(safe_box)
            tile_mask_safe = mask_region & safe_region
            has_error_in_center = not tile_mask_safe.is_empty()

            # --- 3 STATES LOGIC ---
            if has_error_in_center:
                is_dirty = True
            elif not has_error_anywhere:
                is_dirty = False
            else:
                discard_count += 1
                continue

            # --- FIX: DATA BALANCING (The Valve) ---
            # Don't save infinite clean tiles. Keep them balanced with dirty tiles.
            # This drastically reduces runtime and prevents dataset imbalance.
            if not is_dirty and clean_count > dirty_count + 100:
                continue
            # ---------------------------------------

            tile_metal = metal_region & tile_region
            if tile_metal.is_empty():
                continue 

            img = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE), 0)
            draw = ImageDraw.Draw(img)

            for poly in tile_metal.each():
                shifted_poly = poly.moved(-x, -y)
                pts = [(pt.x * scale_factor, pt.y * scale_factor) for pt in shifted_poly.each_point_hull()]
                if len(pts) >= 3:
                    draw.polygon(pts, fill=1) 

            matrix = np.array(img, dtype=np.uint8)
            matrix = np.flipud(matrix)

            metal_density = np.sum(matrix) / (IMAGE_SIZE * IMAGE_SIZE)
            if metal_density < 0.03 or metal_density > 0.85:
                continue

            if is_dirty:
                np.save(f"{OUTPUT_DIR}/dirty/tile_{x}_{y}.npy", matrix)
                dirty_count += 1
            else:
                np.save(f"{OUTPUT_DIR}/clean/tile_{x}_{y}.npy", matrix)
                clean_count += 1

    print(f"Total Dirty: {dirty_count} | Total Clean: {clean_count} | Discarded Edge Errors: {discard_count}")
    return dirty_count, clean_count

if __name__ == "__main__":
    generate_dataset()