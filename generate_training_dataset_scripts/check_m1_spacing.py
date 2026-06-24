import klayout.db as db
import os
import sys

# --- Sky130 Constants ---
M1_LAYER_NUM   = 68
M1_DATATYPE    = 20
M1_2_SPACING_NM = 140   # minimum metal1-to-metal1 spacing (rule m1.2)

def check_m1_spacing(input_gds: str, output_markers_gds: str = None):
    """
    Checks a GDS file for Sky130 rule m1.2:
        Spacing of metal1 to metal1 >= 0.140 µm

    Parameters
    ----------
    input_gds          : path to the input GDS/OAS file
    output_markers_gds : optional path to write a GDS with violation markers.
                         If None, no output file is produced.

    Returns
    -------
    violations : list of dicts, each containing:
        - 'spacing_um'  : actual spacing in µm
        - 'edge1'       : (x1,y1,x2,y2) of the first offending edge (µm)
        - 'edge2'       : (x1,y1,x2,y2) of the second offending edge (µm)
        - 'midpoint'    : (mx, my) midpoint between the two edges (µm)
    """
    if not os.path.exists(input_gds):
        print(f"[ERROR] File not found: {input_gds}")
        return []

    # ------------------------------------------------------------------ load
    print(f"[*] Loading: {input_gds}")
    layout = db.Layout()
    layout.read(input_gds)
    top_cell = layout.top_cell()
    dbu = layout.dbu          # database unit  (usually 0.001 µm per unit)

    m1_idx = layout.find_layer(M1_LAYER_NUM, M1_DATATYPE)
    if m1_idx is None:
        print(f"[ERROR] Layer {M1_LAYER_NUM}/{M1_DATATYPE} (met1) not found in {input_gds}")
        return []

    # ------------------------------------------------ flatten hierarchy → Region
    print("[*] Extracting metal1 geometry (flattening hierarchy)...")
    m1_region = db.Region()
    shapes_iter = top_cell.begin_shapes_rec(m1_idx)
    while not shapes_iter.at_end():
        shape = shapes_iter.shape()
        m1_region.insert(shape.polygon.transformed(shapes_iter.itrans()))
        shapes_iter.next()
    m1_region.merge()

    total_shapes = m1_region.count()
    print(f"[*] Found {total_shapes} metal1 polygon(s) after merge")

    # ---------------------------------------- spacing check (rule m1.2)
    # space_check(distance_dbu) returns EdgePairs where gap < distance
    print(f"[*] Running spacing check (min = {M1_2_SPACING_NM} nm) ...")
    violations_ep = m1_region.space_check(M1_2_SPACING_NM)

    n_violations = violations_ep.count()
    print(f"[*] Found {n_violations} violation edge-pair(s)")

    # ---------------------------------------- collect results
    violations = []
    for ep in violations_ep.each():
        e1 = ep.first
        e2 = ep.second

        def edge_to_um(edge):
            return (
                round(edge.p1.x * dbu, 4),
                round(edge.p1.y * dbu, 4),
                round(edge.p2.x * dbu, 4),
                round(edge.p2.y * dbu, 4),
            )

        e1_um = edge_to_um(e1)
        e2_um = edge_to_um(e2)

        # midpoint between the two edges (average of all four endpoints)
        mx = round((e1_um[0] + e1_um[2] + e2_um[0] + e2_um[2]) / 4, 4)
        my = round((e1_um[1] + e1_um[3] + e2_um[1] + e2_um[3]) / 4, 4)

        # actual spacing = distance between the closest points of the two edges
        spacing_nm = ep.distance()                   # in dbu units
        spacing_um = round(spacing_nm * dbu, 4)

        violations.append({
            "spacing_um": spacing_um,
            "edge1":      e1_um,
            "edge2":      e2_um,
            "midpoint":   (mx, my),
        })

    # ---------------------------------------- print report
    print()
    print("=" * 60)
    print(f"  DRC Report - Rule m1.2 (min spacing {M1_2_SPACING_NM} nm)")
    print(f"  File : {os.path.basename(input_gds)}")
    print("=" * 60)
    if not violations:
        print("  No violations found. Layout is clean.")
    else:
        for i, v in enumerate(violations, 1):
            mx, my = v["midpoint"]
            print(f"  [{i:>4}]  spacing = {v['spacing_um']:.4f} um  |  location ~ ({mx}, {my}) um")
            print(f"          edge1: ({v['edge1'][0]}, {v['edge1'][1]}) -> ({v['edge1'][2]}, {v['edge1'][3]}) um")
            print(f"          edge2: ({v['edge2'][0]}, {v['edge2'][1]}) -> ({v['edge2'][2]}, {v['edge2'][3]}) um")
    print("=" * 60)
    print(f"  Total violations: {n_violations}")
    print("=" * 60)

    # ---------------------------------------- optional: write markers GDS
    if output_markers_gds and violations:
        _write_violation_markers(layout, violations, output_markers_gds, dbu)

    return violations


def _write_violation_markers(source_layout, violations, output_path, dbu):
    """
    Writes a GDS file that contains the original layout *plus* small cross
    markers on layer 255/0 at every violation midpoint, for easy visual
    inspection in KLayout.
    """
    MARKER_LAYER = 255
    MARKER_DT    = 0
    CROSS_SIZE   = int(0.2 / dbu)   # 0.2 µm cross arm length in dbu

    out_ly = db.Layout()
    out_ly.dbu = dbu
    top = out_ly.create_cell("DRC_VIOLATIONS")
    mk_layer = out_ly.layer(MARKER_LAYER, MARKER_DT)

    for v in violations:
        mx_dbu = int(v["midpoint"][0] / dbu)
        my_dbu = int(v["midpoint"][1] / dbu)
        # horizontal bar
        top.shapes(mk_layer).insert(
            db.Box(mx_dbu - CROSS_SIZE, my_dbu - CROSS_SIZE // 5,
                   mx_dbu + CROSS_SIZE, my_dbu + CROSS_SIZE // 5))
        # vertical bar
        top.shapes(mk_layer).insert(
            db.Box(mx_dbu - CROSS_SIZE // 5, my_dbu - CROSS_SIZE,
                   mx_dbu + CROSS_SIZE // 5, my_dbu + CROSS_SIZE))

    out_ly.write(output_path)
    print(f"[*] Violation markers written to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_m1_spacing.py <input.gds> [output_markers.gds]")
        print("Example: python check_m1_spacing.py layout.gds violations.gds")
        sys.exit(1)

    input_file  = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    check_m1_spacing(input_file, output_file)
