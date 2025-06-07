from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from skimage.transform import resize

class UniversalGrad(nn.Module):
    def __init__(self, model, target_layer_name):
        super(UniversalGrad, self).__init__()
        
        self.model = model
        self.gradients = None
        self.activations = None
        self.modules = dict(model.named_modules())
        self.target_layer = self.modules[target_layer_name]
        self.forward_hook = self.target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = self.target_layer.register_full_backward_hook(self._save_gradient)

            
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        if grad_output[0] is not None:
            self.gradients = grad_output[0].detach()
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def forward(self, x):
        self.gradients = None
        self.activations = None
        output = self.model(x)
        return output
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        self.gradients = None
        self.activations = None
        
        with torch.no_grad():
            _ = self.model(x)
        
        return self.activations
    
    def __del__(self):
        if hasattr(self, 'forward_hook'):
            self.forward_hook.remove()
        if hasattr(self, 'backward_hook'):
            self.backward_hook.remove()


def gradCAM(model, loader, device):

    heatmaps = []
    images = []
    predicted_labels = []
    real_labels = []

    batch = next(iter(loader))

    for image, label in zip(batch['image'], batch['diagnosis']):
        
        img = image.to(device).requires_grad_()
        labl = label.to(device).argmax(dim=0).item()
        real_labels.append(labl)

        if img.ndim == 3:
            img = img.unsqueeze(0)

        # 1. Get prediction
        pred = model(img)
        class_idx = pred.argmax(dim=1).item()
        predicted_labels.append(class_idx)

        # 2. Backward pass
        model.zero_grad()
        pred[0, class_idx].backward()

        # 3. Pull gradients
        gradients = model.get_activations_gradient()
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # 4. Get activations
        activations = model.get_activations(img).detach()
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze().cpu()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= torch.max(heatmap)
        heatmaps.append(heatmap)

        # Convert image tensor to numpy
        img_np = img.detach().squeeze().permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)  # If normalized

        images.append(img_np)

    return heatmaps, images, predicted_labels, real_labels


def overlay_heatmap(img, heatmap, alpha=0.4, colormap='jet'):

    if img.max() > 1.0:
        img = img / 255.0  

    cmap = matplotlib.colormaps[colormap]
    colored_heatmap = cmap(heatmap)[:, :, :3]  
    
    overlay = (1 - alpha) * img + alpha * colored_heatmap
    overlay = np.clip(overlay, 0, 1)

    return overlay

def visualize_gradcams_with_colorbars(images, heatmaps, preds, labels, conditions, max_rows=4):
    
    cols = 3
    max_images = max_rows * cols
    n_images = min(len(images), max_images)
    rows = (n_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for idx, (img, heatmap, pred, label) in enumerate(zip(images, heatmaps, preds, labels)):
        
        if idx >= max_images:
            break

        heatmap = heatmap.squeeze()  
        if heatmap.ndim > 2:
            heatmap = heatmap[:, :, 0]  
        
        heatmap = heatmap.detach().cpu().numpy()  
        heatmap = heatmap - np.min(heatmap)
        heatmap = heatmap / np.max(heatmap + 1e-8)

        heatmap = resize(heatmap, (224, 224), anti_aliasing=True)
       
        overlay = overlay_heatmap(img, heatmap)

        axes[idx].imshow(overlay)
        axes[idx].set_title(f'Condition: {conditions[label]}, Predicted: {conditions[pred]}')
        axes[idx].axis('off')
    
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
