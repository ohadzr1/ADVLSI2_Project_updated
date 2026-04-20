import klayout.db as db
import os

def final_extract_m1(input_file):
    # Load the layout
    ly = db.Layout()
    ly.read(input_file)
    top_cell = ly.top_cell()
    
    # Sky130 Metal 1 definition (Layer 68, Datatype 20)
    target_layer, target_datatype = 68, 20
    m1_index = ly.find_layer(target_layer, target_datatype)
    
    if m1_index is None:
        print(f"Layer {target_layer}/{target_datatype} not found")
        return

    m1_region = db.Region()
    print("Extracting geometry from hierarchy...")

    # Recursive traversal of the hierarchy
    shapes_iter = top_cell.begin_shapes_rec(m1_index)
    while not shapes_iter.at_end():
        shape = shapes_iter.shape()
        # Use .polygon property and .transformed() method to flatten coordinates
        m1_region.insert(shape.polygon.transformed(shapes_iter.itrans()))
        shapes_iter.next()

    # Merge overlapping polygons to create continuous shapes
    m1_region.merge()

    # Initialize the output layout
    out_ly = db.Layout()
    out_ly.dbu = ly.dbu
    out_top = out_ly.create_cell("CLEAN_M1")
    out_layer = out_ly.layer(target_layer, target_datatype)
    out_top.shapes(out_layer).insert(m1_region)
    
    # Generate output filename based on original input name
    file_base = os.path.splitext(os.path.basename(input_file))[0]
    output_gds = f"{file_base}_M1.gds"
    
    # Write the result to disk
    out_ly.write(output_gds)
    print(f"Success! Clean file created: {output_gds}")

if __name__ == "__main__":
    # Ensure the input filename is correct
    final_extract_m1("tt_um_yen.oas")