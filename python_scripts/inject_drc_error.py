import klayout.db as db
import random
import os
import math

INPUT_FILE_NAME = "tt_um_yen_M1.gds"
M1_LAYER_NUM = 68
M1_DATATYPE = 20
MARKING_LAYER_NUM = 255
MARKING_DATATYPE = 0

MIN_WIDTH = 140
M1_2_SPACING = 140   
M1_3_SPACING = 280   
HUGE_THRESHOLD = 3000 
MIN_AREA = 83000     
GRID_STEP = 5        

# FIX: Minimum gap to prevent polygons from merging due to pixel quantization
MIN_VISIBLE_GAP = 40 

def snap_to_grid(value, grid=GRID_STEP):
    return int(round(value / grid) * grid)

def inject_isolated_m1_2_errors(input_file, num_errors=500): 
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    layout = db.Layout()
    layout.read(input_file)
    top_cell = layout.top_cell()
    
    m1_idx = layout.find_layer(M1_LAYER_NUM, M1_DATATYPE)
    marking_idx = layout.layer(MARKING_LAYER_NUM, MARKING_DATATYPE)
    
    main_region = db.Region(top_cell.shapes(m1_idx))
    main_region.merge()
    
    huge_metal = main_region.sized(-HUGE_THRESHOLD // 2).sized(HUGE_THRESHOLD // 2)
    regular_metal = main_region - huge_metal
    
    bbox = main_region.bbox()
    errors_injected = 0
    attempts = 0
    max_attempts = num_errors * 2000 

    print(f"Processing: {input_file} - Injecting {num_errors} errors...")

    while errors_injected < num_errors and attempts < max_attempts:
        attempts += 1
        
        w = snap_to_grid(random.randrange(140, 400)) 
        min_l_required = math.ceil(MIN_AREA / w) 
        lower_bound = snap_to_grid(min_l_required) 
        upper_bound = snap_to_grid(lower_bound + random.randrange(10, 200))
        
        l = snap_to_grid(random.randrange(lower_bound, max(lower_bound + GRID_STEP, upper_bound)))
        rect_w, rect_l = (w, l) if random.choice([True, False]) else (l, w)

        x = snap_to_grid(random.randrange(bbox.left, bbox.right - rect_w))
        y = snap_to_grid(random.randrange(bbox.bottom, bbox.top - rect_l))
        
        error_box = db.Box(x, y, x + rect_w, y + rect_l)
        error_region = db.Region(error_box)
        
        # FIX: Ensure a minimum gap of 40nm from any existing metal to prevent merging
        if not (main_region & error_region.sized(MIN_VISIBLE_GAP)).is_empty(): continue
        
        huge_safety_halo = error_region.sized(M1_3_SPACING)
        if not (huge_metal & huge_safety_halo).is_empty(): continue 
            
        regular_violation_halo = error_region.sized(M1_2_SPACING)
        if (regular_metal & regular_violation_halo).is_empty(): continue 
            
        top_cell.shapes(m1_idx).insert(error_box)
        top_cell.shapes(marking_idx).insert(error_box)
        
        main_region.insert(error_box)
        main_region.merge()
        
        errors_injected += 1

    base, ext = os.path.splitext(input_file)
    output_name = f"{base}_m1_2_Marked{ext}"
    layout.write(output_name)
    print("--------------------------------------------------")
    print(f"Process completed. Created: {output_name}")
    print(f"Total isolated violations: {errors_injected}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    inject_isolated_m1_2_errors(INPUT_FILE_NAME, num_errors=500)