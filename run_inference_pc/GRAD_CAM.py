import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — saves files instead of opening windows
import matplotlib.pyplot as plt
import cv2


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.eval()

        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))

        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]

        heatmap = np.maximum(heatmap, 0)
        heatmap = cv2.resize(heatmap, (input_tensor.shape[3], input_tensor.shape[2]))
        heatmap = heatmap - np.min(heatmap)
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        return heatmap


def generate_gradcam(matrix, model, device, save_path, class_idx=1):
    """
    Run Grad-CAM on a numpy tile array and save the result as a PNG.

    Args:
        matrix:     2-D float32 numpy array (200x200) — the tile to analyse.
        model:      Loaded NCSU_DRCNN model (already on device).
        device:     torch.device to run on.
        save_path:  Full path to save the output PNG image.
        class_idx:  1 = violation class (default), 0 = clean class.
    """
    # Grad-CAM requires gradients — explicitly enable them even if called
    # inside a torch.no_grad() context in the caller.
    with torch.enable_grad():
        input_tensor = torch.from_numpy(matrix).unsqueeze(0).unsqueeze(0).to(device)
        input_tensor.requires_grad = True

        grad_cam = GradCAM(model, model.conv4)
        heatmap = grad_cam.generate_heatmap(input_tensor, class_idx=class_idx)

    # Prepare original layout image for display
    img_display = matrix.copy()
    img_display = (img_display - np.min(img_display)) / max(np.max(img_display) - np.min(img_display), 1e-8)
    img_display = np.uint8(255 * img_display)
    img_display = 255 - img_display  # Invert: metal = dark, background = white

    # Colorize heatmap and superimpose
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed = heatmap_colored * 0.5 + cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB) * 0.5
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_display, cmap='gray')
    axes[0].set_title("Original Layout")
    axes[0].axis('off')

    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis('off')

    axes[2].imshow(superimposed)
    axes[2].set_title("Superimposed (Violation Area)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
