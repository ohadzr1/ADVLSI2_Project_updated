import subprocess
import os

# --- CONFIGURATION ---
# The path to your KLayout executable (if it's in your system PATH, just "klayout" works)
KLAYOUT_CMD = r"C:\Users\ohadz\AppData\Roaming\KLayout\klayout_app.exe"

# The actual DRC script file for sky130 (update this to your actual .lydrc or .drc file name)
DRC_SCRIPT = "run_drc_full.lydrc" 

def run_full_drc(input_gds, output_rdb):
    """
    Runs KLayout in batch mode (no GUI) to execute a DRC script.
    It passes the input GDS and the desired output RDB file as variables.
    """
    if not os.path.exists(DRC_SCRIPT):
        print(f"Error: DRC script '{DRC_SCRIPT}' not found!")
        print("Please ensure the sky130 DRC rule deck is in the folder.")
        return False

    print(f"[*] Starting KLayout DRC Engine in batch mode...")
    
    # Constructing the CLI command
    # -b: Batch mode (no UI)
    # -r: Run the specified script
    # -rd: Pass variables to the script (input and report)
    command = [
        KLAYOUT_CMD,
        "-b", 
        "-r", DRC_SCRIPT,
        "-rd", f"input={input_gds}",
        "-rd", f"report={output_rdb}"
    ]

    try:
        # Run the command and wait for it to finish
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"[*] DRC completed successfully. Report saved to: {output_rdb}")
        return True
        
    except subprocess.CalledProcessError as e:
        print("!!! DRC Execution Failed !!!")
        print("Error output from KLayout:")
        print(e.stderr)
        return False

if __name__ == "__main__":
    run_full_drc("tt_um_yen_M1_m1_2_Marked.gds", "sky130_drc.txt")