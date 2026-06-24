import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks to extract activations and gradients during forward/backward passes
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.eval()

        # Forward pass to get the model's predictions
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        # Backward pass to calculate gradients for the target class
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Retrieve the computed gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        # Calculate the channel weights based on the mean of the gradients
        weights = np.mean(gradients, axis=(1, 2))

        # Generate the heatmap by computing the weighted sum of the activations
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]

        # Apply ReLU to keep only features that have a positive influence on the class, and normalize
        heatmap = np.maximum(heatmap, 0)
        heatmap = cv2.resize(heatmap, (input_tensor.shape[3], input_tensor.shape[2]))
        heatmap = heatmap - np.min(heatmap)
        heatmap = heatmap / np.max(heatmap)
        return heatmap

def show_gradcam_on_layout(npy_path, model, class_idx=1): # class_idx=1 represents the "dirty/violation" class
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load the layout numpy array
    matrix = np.load(npy_path).astype(np.float32)
    input_tensor = torch.from_numpy(matrix).unsqueeze(0).unsqueeze(0).to(device)
    input_tensor.requires_grad = True # Required to compute gradients for Grad-CAM

    # Initialize GradCAM pointing to the deepest convolutional layer (conv4)
    grad_cam = GradCAM(model, model.conv4)

    # Generate the heatmap for the specified violation class
    heatmap = grad_cam.generate_heatmap(input_tensor, class_idx=class_idx)

    # Prepare the original layout image for display (normalization)
    img_display = matrix
    img_display = (img_display - np.min(img_display)) / (np.max(img_display) - np.min(img_display))
    img_display = np.uint8(255 * img_display)

    # Invert colors: Ensure metal polygons and background match the expected visual representation
    img_display = 255 - img_display

    # Convert the heatmap to RGB format (Red indicates "hot" areas with high influence)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Superimpose the heatmap onto the original layout image
    # Weights (0.5 and 0.5) are balanced to keep the underlying layout geometry visible
    superimposed_img = heatmap_colored * 0.5 + cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB) * 0.5
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    # Plotting the three stages: Original, Heatmap, and Superimposed result
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original Layout")
    plt.imshow(img_display, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Grad-CAM Heatmap")
    plt.imshow(heatmap, cmap='jet')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Superimposed (Violation Area)")
    plt.imshow(superimposed_img)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# --- Example Usage ---
# Make sure to run this after your model is loaded/trained
# sample_dirty_layout = 'data/dirty/tile_example_123.npy'
# show_gradcam_on_layout(sample_dirty_layout, model, class_idx=1)


# Make sure your model is loaded with the trained weights
model = NCSU_DRCNN()
model.load_state_dict(torch.load('ncsu_drcnn_weights.pth', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

# Point to a specific tile you know has a violation
sample_dirty_layout = 'data/dirty/tt_um_8_bit_cpu_tile_74360_60880.npy'

# Run the visualization!
show_gradcam_on_layout(sample_dirty_layout, model, class_idx=1)