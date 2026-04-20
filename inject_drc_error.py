import klayout.db as db
import random
import os
import math

# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_FILE_NAME = "tt_um_yen_M1.gds"
M1_LAYER_NUM = 68
M1_DATATYPE = 20
MARKING_LAYER_NUM = 255
MARKING_DATATYPE = 0

# Sky130 DRC Rules (in nanometers)
MIN_WIDTH = 140
M1_2_SPACING = 140   
M1_3_SPACING = 280   
HUGE_THRESHOLD = 3000 
MIN_AREA = 83000     
GRID_STEP = 5        

# Maximum dimension allowed for injected rectangles
MAX_RECT_DIM = 1500
# =============================================================================

def snap_to_grid(value, grid=GRID_STEP):
    """Aligns values to the manufacturing grid."""
    return int(round(value / grid) * grid)

def inject_isolated_m1_2_errors(input_file, num_errors=100):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    layout = db.Layout()
    layout.read(input_file)
    top_cell = layout.top_cell()
    
    m1_idx = layout.find_layer(M1_LAYER_NUM, M1_DATATYPE)
    marking_idx = layout.layer(MARKING_LAYER_NUM, MARKING_DATATYPE)
    
    if m1_idx is None:
        print(f"Error: Layer {M1_LAYER_NUM}/{M1_DATATYPE} not found.")
        return

    # Create a merged region of all existing Metal 1
    main_region = db.Region(top_cell.shapes(m1_idx))
    main_region.merge()
    
    # Identify Huge Metal (width > 3um) for rule m1.3 isolation
    huge_metal = main_region.sized(-HUGE_THRESHOLD // 2).sized(HUGE_THRESHOLD // 2)
    regular_metal = main_region - huge_metal
    
    bbox = main_region.bbox()
    errors_injected = 0
    attempts = 0
    max_attempts = num_errors * 1000 

    print(f"Processing: {input_file}")

    while errors_injected < num_errors and attempts < max_attempts:
        attempts += 1
        
        # 1. Generate width
        w = snap_to_grid(random.randrange(MIN_WIDTH, 600))
        
        # 2. Calculate minimum length to satisfy MIN_AREA
        # min_l = Area / width
        min_l_required = math.ceil(MIN_AREA / w)
        
        # 3. Establish valid range for length
        # The lower bound must be the greater of MIN_WIDTH or min_l_required
        lower_bound = snap_to_grid(max(MIN_WIDTH, min_l_required))
        # The upper bound must be strictly greater than the lower bound
        upper_bound = snap_to_grid(max(lower_bound + GRID_STEP, MAX_RECT_DIM))
        
        l = snap_to_grid(random.randrange(lower_bound, upper_bound))
        
        # Randomize orientation
        rect_w, rect_l = (w, l) if random.choice([True, False]) else (l, w)

        x = snap_to_grid(random.randrange(bbox.left, bbox.right - rect_w))
        y = snap_to_grid(random.randrange(bbox.bottom, bbox.top - rect_l))
        
        error_box = db.Box(x, y, x + rect_w, y + rect_l)
        error_region = db.Region(error_box)
        
        # --- DRC Isolation Logic ---
        
        # Ensure no short circuit (does not overlap existing metal)
        if not (main_region & error_region).is_empty():
            continue
            
        # Ensure no m1.3 violation (must be > 280nm from huge metal)
        huge_safety_halo = error_region.sized(M1_3_SPACING)
        if not (huge_metal & huge_safety_halo).is_empty():
            continue 
            
        # Ensure m1.2 violation (must be <= 140nm from regular metal)
        regular_violation_halo = error_region.sized(M1_2_SPACING)
        if (regular_metal & regular_violation_halo).is_empty():
            continue 
            
        # Injection confirmed
        top_cell.shapes(m1_idx).insert(error_box)
        top_cell.shapes(marking_idx).insert(error_box)
        
        # Update region for next iteration
        main_region.insert(error_box)
        main_region.merge()
        
        errors_injected += 1
        if errors_injected % 10 == 0:
            print(f"Status: {errors_injected} errors placed.")

    # File naming logic to prevent overwriting and handle file extensions
    base, ext = os.path.splitext(input_file)
    output_name = f"{base}_m1_2_Marked{ext}"
    
    layout.write(output_name)
    print("--------------------------------------------------")
    print(f"Process completed. Created: {output_name}")
    print(f"Total isolated violations: {errors_injected}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    inject_isolated_m1_2_errors(INPUT_FILE_NAME, num_errors=100)