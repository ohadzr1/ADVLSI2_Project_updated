
# Inference - model excecution on inference dataset

import os
import numpy as np
import torch
import time
import zipfile
import io

# --- CONFIGURATION ---
# Point directly to the ZIP file shown in your Colab root directory
ZIP_PATH ="/content/inference_dataset.zip"
CONFIDENCE_THRESHOLD = 0.80

def scan_layout_from_zip(model, zip_file_path, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    if not os.path.exists(zip_file_path):
        print(f"[!] Error: Could not find ZIP file at {zip_file_path}")
        return

    print(f"[*] Opening ZIP archive at {zip_file_path}...")

    violations = []
    start_time = time.time()

    # Open the ZIP file directly
    with zipfile.ZipFile(zip_file_path, 'r') as archive:
        # Filter only .npy files from the archive list
        npy_files = [f for f in archive.namelist() if f.endswith('.npy')]

        if not npy_files:
            print("[!] Error: No .npy files found inside the ZIP.")
            return

        print(f"[*] Starting DRC scan on {len(npy_files)} tiles directly from ZIP using {device}...")

        with torch.no_grad():
            for i, filename in enumerate(npy_files):
                # Read the raw bytes from the ZIP into memory
                with archive.open(filename) as f:
                    file_bytes = f.read()
                    # Convert bytes directly to a NumPy array
                    matrix = np.load(io.BytesIO(file_bytes)).astype(np.float32)

                # Format for CNN: (1, 1, 200, 200)
                tensor = torch.from_numpy(matrix).unsqueeze(0).unsqueeze(0).to(device)

                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)
                violation_prob = probs[0][1].item()

                if violation_prob >= threshold:
                    # Extract pure filename without folder paths that might be in the ZIP
                    base_name = os.path.basename(filename)
                    coords = base_name.replace("tile_", "").replace(".npy", "")
                    violations.append((coords, violation_prob))

                if (i + 1) % 2000 == 0:
                    print(f"[>] Scanned {i + 1}/{len(npy_files)} tiles...")

    end_time = time.time()
    print(f"\n[+] Scan complete in {end_time - start_time:.1f} seconds!")

    # --- Print results and save report ---
    print("=" * 40)
    if not violations:
        print("[*] LAYOUT IS CLEAN! No DRC violations found.")
    else:
        print(f"[!] FOUND {len(violations)} POTENTIAL VIOLATIONS!")
        print("=" * 40)

        with open("drc_report.txt", "w") as f:
            f.write("CNN DRC Violations Report\n")
            f.write("=========================\n")
            for coords, prob in violations:
                line = f"Location: {coords} | Confidence: {prob:.2%}"
                print(f"  - {line}")
                f.write(line + "\n")

        print("\n[-] Detailed report saved to 'drc_report.txt' in Colab files.")

# --- 1. Model Initialization ---
model = NCSU_DRCNN()

# --- 2. Load Trained Weights ---
model_weights_path = 'ncsu_drcnn_weights.pth'

model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=True))

# --- 3. Execute Inference ---
scan_layout_from_zip(model, ZIP_PATH, CONFIDENCE_THRESHOLD)
