# Autonomous DRC Dataset Generation Pipeline

This project contains an automated pipeline for generating VLSI layout datasets aimed at training Machine Learning models (like CNNs) for Design Rule Check (DRC) violation detection. 

The pipeline autonomously extracts target layers (e.g., Metal 1) from real layouts, injects synthetic spacing violations, executes the KLayout DRC engine in batch mode, extracts the verified violations, and slices the layout into labeled (clean vs. dirty) grid matrices.

## Features
* **Layer Extraction:** Isolates target layers (e.g., Sky130 M1) from `.gds` or `.oas` layout files.
* **Error Injection:** Synthetically injects realistic, isolated spacing violations into the layout.
* **Automated DRC:** Runs KLayout DRC in batch mode and parses the Report Database (RDB) results.
* **Dataset Generation:** Tiles the layout into `.npy` matrices and balances clean vs. dirty samples.
* **Visualization:** Includes built-in tools to visually inspect the resulting matrices for both training and inference.

## Prerequisites
* **Python 3.x**
* **KLayout** (Ensure the `KLAYOUT_CMD` path in `run_full_drc.py` matches your system installation)
* **Sky130 DRC rule deck** (expected to be located at `sky130_drc_deck/run_drc_full.lydrc`)

### Install Dependencies
You can install the required Python packages using pip:
```bash
pip install numpy matplotlib klayout Pillow
```

## Execution Instructions

### 1. Run the Full Training Pipeline
To execute the complete end-to-end flow (Cleanup -> Extract -> Inject -> DRC -> Mask -> Generate Dataset -> Visualize):
```bash
python run_flow.py
```

### 2. Visualize Training Data (Sanity Check)
If you want to run the dataset visualizer independently without regenerating the data:
```bash
python visualize_dataset_matrices.py
```

### 3. Generate Inference Dataset
To slice an existing, un-injected layout into overlapping tiles for model inference:
```bash
python generate_inference_dataset.py
```

### 4. Visualize Inference Data
To inspect specific layout tiles generated for inference (you can specify target coordinates inside the script):
```bash
python visualize_inference_dataset.py
```
