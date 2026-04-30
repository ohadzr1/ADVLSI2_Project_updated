import os
import shutil
import numpy as np
import klayout.db as db
from PIL import Image, ImageDraw
from extract_m1 import extract_m1

# --- CONFIGURATION (Based on your training parameters) ---
LAYOUT_NAME = "tt_um_yen"
INPUT_GDS = f"real_layouts_tt/{LAYOUT_NAME}.oas"
GENERATED_DIR = f"outputs/{LAYOUT_NAME}/dataset_output"
OUTPUT_DIR = f"outputs/{LAYOUT_NAME}/inference_dataset"
METAL_LAYER = (68, 20)

# Physical window size used during training
PHYSICAL_SIZE = 1600    
# Stride set to 1500nm to provide 100nm overlap and prevent file explosion
STRIDE = 1500           
# Target matrix size for the CNN
IMAGE_SIZE = 200        
# Ratio to convert nm coordinates to pixels (0.125)
SCALE_FACTOR = IMAGE_SIZE / PHYSICAL_SIZE 

def generate_inference_dataset(input_gds=INPUT_GDS, generated_dir=GENERATED_DIR, output_dir=OUTPUT_DIR):
    # 1. Extract Metal 1 layer using your existing function
    print(f"[*] Extracting Metal 1 from {input_gds}...")
    extract_m1(input_gds, generated_dir)
    file_base = os.path.splitext(os.path.basename(input_gds))[0]
    m1_gds = os.path.join(generated_dir, f"{file_base}_M1.gds")

    # 2. Load the layout and get the Metal 1 region
    layout = db.Layout()
    layout.read(m1_gds)
    top_cell = layout.top_cell()
    layer_idx = layout.find_layer(*METAL_LAYER)
    metal_region = db.Region(top_cell.begin_shapes_rec(layer_idx))
    bbox = metal_region.bbox()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 3. Calculate total steps for progress tracking
    x_range = range(bbox.left, bbox.right - PHYSICAL_SIZE, STRIDE)
    y_range = range(bbox.bottom, bbox.top - PHYSICAL_SIZE, STRIDE)
    total_steps = len(x_range) * len(y_range)
    
    print(f"[*] Layout Size: {bbox.width()/1000:.2f}um x {bbox.height()/1000:.2f}um")
    print(f"[*] Strategy: Window={PHYSICAL_SIZE}nm, Stride={STRIDE}nm")
    print(f"[*] Total windows to process: {total_steps}")

    processed_count = 0
    saved_count = 0

    # 4. Tiling Loop
    for x in x_range:
        for y in y_range:
            processed_count += 1
            
            # Define current window and check for metal presence
            tile_box = db.Box(x, y, x + PHYSICAL_SIZE, y + PHYSICAL_SIZE)
            tile_metal = metal_region & db.Region(tile_box)
            
            if not tile_metal.is_empty():
                # Create grayscale image (L mode)
                img = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE), 0)
                draw = ImageDraw.Draw(img)

                for poly in tile_metal.each():
                    # Move polygon to local tile coordinates
                    shifted_poly = poly.moved(-x, -y)
                    # Scale nm to pixels
                    pts = [(pt.x * SCALE_FACTOR, pt.y * SCALE_FACTOR) for pt in shifted_poly.each_point_hull()]
                    if len(pts) >= 3:
                        draw.polygon(pts, fill=1)

                # Convert to numpy and flip for GDS orientation
                matrix = np.array(img, dtype=np.uint8)
                matrix = np.flipud(matrix)

                # Optional: Only save if there is significant metal density
                if np.mean(matrix) > 0.01:
                    file_name = f"tile_x{x}_y{y}.npy"
                    np.save(os.path.join(output_dir, file_name), matrix)
                    saved_count += 1

            # Print progress percentage every 500 tiles
            if processed_count % 500 == 0 or processed_count == total_steps:
                progress = (processed_count / total_steps) * 100
                print(f"[>] Progress: {progress:.1f}% | Tiles Saved: {saved_count}")

    zip_path = shutil.make_archive(output_dir, "zip", root_dir=output_dir)
    print(f"[*] Created archive: {zip_path}")
    print(f"\n Done! Processed {total_steps} areas. Saved {saved_count} tiles to '{output_dir}'.")

if __name__ == "__main__":
    generate_inference_dataset()