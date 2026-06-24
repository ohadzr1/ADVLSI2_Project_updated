import subprocess
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import SKY130_DRC_SCRIPT, find_klayout_executable


def run_full_drc(input_gds, output_rdb):
    """
    Runs KLayout in batch mode (no GUI) to execute a DRC script.
    It passes the input GDS and the desired output RDB file as variables.
    """
    drc_script = SKY130_DRC_SCRIPT
    klayout_cmd = find_klayout_executable()

    if not drc_script.exists():
        print(f"Error: DRC script '{drc_script}' not found!")
        print("Please ensure the sky130 DRC rule deck is in the project folder.")
        return False

    abs_input = str(Path(input_gds).resolve())
    abs_output = str(Path(output_rdb).resolve())

    print(f"[*] Starting KLayout DRC Engine in batch mode...")
    print(f"[*] Using KLayout executable: {klayout_cmd}")

    command = [
        klayout_cmd,
        "-b",
        "-r", str(drc_script),
        "-rd", f"input={abs_input}",
        "-rd", f"report={abs_output}",
    ]

    try:
        subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"[*] DRC completed successfully. Report saved to: {output_rdb}")
        return True

    except subprocess.CalledProcessError as e:
        print("!!! DRC Execution Failed !!!")
        print("Error output from KLayout:")
        print(e.stderr)
        return False


if __name__ == "__main__":
    from project_paths import injected_m1_gds, drc_report_path

    layout_name = "tt_um_yen"
    run_full_drc(injected_m1_gds(layout_name), drc_report_path(layout_name))
