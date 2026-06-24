import torch
import numpy as np
import cv2


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Register forward/backward hooks once at construction and reuse the same
    instance for all tiles — avoids re-registering hooks on every call.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=1):
        """One forward+backward pass → normalised heatmap (H x W, float32)."""
        self.model.eval()
        output = self.model(input_tensor)

        self.model.zero_grad()
        output[0, class_idx].backward()

        gradients   = self.gradients.cpu().data.numpy()[0]    # (C, h, w)
        activations = self.activations.cpu().data.numpy()[0]  # (C, h, w)

        weights = np.mean(gradients, axis=(1, 2))             # (C,)
        heatmap = np.einsum('c,chw->hw', weights, activations).astype(np.float32)

        heatmap = np.maximum(heatmap, 0)
        heatmap = cv2.resize(heatmap, (input_tensor.shape[3], input_tensor.shape[2]))
        heatmap -= heatmap.min()
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap


def generate_gradcam(matrix, grad_cam, device, save_path, class_idx=1):
    """
    Run Grad-CAM on a single tile and save a 3-panel PNG using cv2 (no matplotlib).

    Args:
        matrix:    2-D float32 numpy array (H x W) — the tile to analyse.
        grad_cam:  Pre-constructed GradCAM instance (reused across all calls).
        device:    torch.device to run on.
        save_path: Full path for the output PNG.
        class_idx: 1 = violation class (default), 0 = clean class.
    """
    # Grad-CAM needs gradients — re-enable them even if caller uses no_grad.
    with torch.enable_grad():
        input_tensor = (torch.from_numpy(matrix)
                        .unsqueeze(0).unsqueeze(0)
                        .to(device))
        input_tensor.requires_grad_(True)
        heatmap = grad_cam.generate_heatmap(input_tensor, class_idx=class_idx)

    # --- Build 3-panel image with numpy/cv2 only ---

    # Panel 1: original layout (inverted — metal dark on white background)
    orig = matrix.copy()
    orig = (orig - orig.min()) / max(orig.max() - orig.min(), 1e-8)
    orig_u8  = np.uint8(255 * (1.0 - orig))
    orig_bgr = cv2.cvtColor(orig_u8, cv2.COLOR_GRAY2BGR)

    # Panel 2: Grad-CAM heatmap (JET colormap)
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # Panel 3: 50/50 blend of heatmap and original
    superimposed = np.clip(
        heatmap_bgr.astype(np.float32) * 0.5 + orig_bgr.astype(np.float32) * 0.5,
        0, 255
    ).astype(np.uint8)

    # Add a thin label bar above each panel
    h, w = orig_bgr.shape[:2]

    def _labeled(img, text):
        bar = np.full((28, w, 3), 240, dtype=np.uint8)
        cv2.putText(bar, text, (4, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (30, 30, 30), 1, cv2.LINE_AA)
        return np.vstack([bar, img])

    panel = np.hstack([
        _labeled(orig_bgr,     "Original Layout"),
        _labeled(heatmap_bgr,  "Grad-CAM Heatmap"),
        _labeled(superimposed, "Superimposed"),
    ])

    cv2.imwrite(save_path, panel)
    
    return heatmap
