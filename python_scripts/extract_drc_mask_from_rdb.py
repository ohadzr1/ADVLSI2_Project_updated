import klayout.db as db
import klayout.rdb as rdb

# --- CONFIGURATION ---
RDB_FILE = "sky130_drc.txt"                    
OUTPUT_MASK_FILE = "drc_mask_layer_255.oas"      
MASK_LAYER = (255, 0)                          
TARGET_NAME = "m1.2"                     

def extract_drc_mask_from_rdb():
    # 1. Create a new, empty layout
    layout = db.Layout()
    # Set the Database Unit (DBU) to 1nm to match the precision of the XML
    layout.dbu = 0.001 
    top_cell = layout.create_cell("TOP")
    mask_idx = layout.layer(*MASK_LAYER)
    
    # 2. Load the DRC database (RDB)
    rdb_data = rdb.ReportDatabase("DRC_Database")
    rdb_data.load(RDB_FILE)
    
    print("Processing all DRC items...")
    
    violation_count = 0
    dbu = layout.dbu
    
    # 3. Iterate through all items in the database
    for item in rdb_data.each_item():
        
        # 4. Access the category name
        item_category = rdb_data.category_by_id(item.category_id())
        category_name = item_category.name()
        
        # 5. Filter for our target category (m1.2)
        clean_name = category_name.replace("'", "").strip()
        
        if clean_name == TARGET_NAME:
            for value in item.each_value():
                if value.is_edge_pair():
                    # Extract the micron-based edge pair (DEdgePair)
                    d_edge_pair = value.edge_pair()
                    # Convert to integer-based edge pair (EdgePair) using layout DBU
                    i_edge_pair = d_edge_pair.to_itype(dbu)
                    
                    # FIX: Pass '0' to .polygon() to create the exact gap polygon
                    # without any additional sizing/enlargement.
                    top_cell.shapes(mask_idx).insert(i_edge_pair.polygon(0))
                    violation_count += 1
                    
                elif value.is_box():
                    d_box = value.box()
                    top_cell.shapes(mask_idx).insert(d_box.to_itype(dbu))
                    violation_count += 1
                    
                elif value.is_polygon():
                    d_poly = value.polygon()
                    top_cell.shapes(mask_idx).insert(d_poly.to_itype(dbu))
                    violation_count += 1

    # 6. Save the mask output
    if violation_count > 0:
        layout.write(OUTPUT_MASK_FILE)
        print(f"\nSuccess! Extracted {violation_count} violations from '{TARGET_NAME}'.")
        print(f"Mask file saved: {OUTPUT_MASK_FILE}")
    else:
        print(f"\nNo violations found for category '{TARGET_NAME}'.")
    
    return violation_count

if __name__ == "__main__":
    extract_drc_mask_from_rdb()