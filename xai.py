from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


class resNet(nn.Module):
    def __init__(self, model):
        super(resNet, self).__init__()

        self.res = model
       
        self.seq = nn.Sequential(
            self.res.conv1,
            self.res.bn1,
            self.res.relu,
            self.res.maxpool,
            self.res.layer1,
            self.res.layer2,
            self.res.layer3,
            self.res.layer4[:-1],
            self.res.layer4[-1].conv1,
            self.res.layer4[-1].bn1,
            self.res.layer4[-1].conv2,
            self.res.layer4[-1].bn2,
            self.res.layer4[-1].conv3
        )

        self.bottom = nn.Sequential(
            self.res.layer4[-1].bn3,
            self.res.layer4[-1].relu,
            self.res.avgpool
        )

        self.fc = self.res.fc

        self.gradients = None


    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.seq(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        x = self.bottom(x)
        x = x.view((1, -1))
        x = self.fc(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.seq(x)
    


def gradCAM(model, loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        # pull the gradients out of the model
        gradients = model.get_activations_gradient()

        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # get the activations of the last convolutional layer
        activations = model.get_activations(img).detach()

        # weight the channels by corresponding gradients
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()

        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        heatmap = np.maximum(heatmap, 0)

        # normalize the heatmap
        heatmap /= torch.max(heatmap)

        heatmaps.append(heatmap)

        # Convert image tensor to numpy
        img_np = img.detach().squeeze().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)  # If normalized

        images.append(img_np)

    return heatmaps, images, predicted_labels, real_labels




def overlay_heatmap_on_image(img, heatmap, alpha=0.4, colormap='jet'):

    if img.max() > 1.0:
        img = img / 255.0  # Convert to [0, 1]

    cmap = matplotlib.colormaps[colormap]
    colored_heatmap = cmap(heatmap)[:, :, :3]  # Drop alpha channel
    
    overlay = (1 - alpha) * img + alpha * colored_heatmap
    overlay = np.clip(overlay, 0, 1)

    return overlay

def visualize_gradcams_with_colorbars(images, heatmaps, preds, labels, conditions):
    cols= 3

    n_images = len(images)
    rows = (n_images + cols - 1) // cols  

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    
    if n_images == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten()
    elif cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for idx, (img, heatmap, pred, label) in enumerate(zip(images, heatmaps, preds, labels)):

        heatmap = heatmap.squeeze()  # Remove any singleton dimensions
        if heatmap.ndim > 2:
            heatmap = heatmap[:, :, 0]  # Take first channel if multiple exist
        
        heatmap = heatmap.detach().cpu().numpy()  # Safely convert to numpy
        heatmap = heatmap - np.min(heatmap)
        heatmap = heatmap / np.max(heatmap + 1e-8)

        from skimage.transform import resize
        heatmap = resize(heatmap, (224, 224), anti_aliasing=True)
       
        overlay = overlay_heatmap_on_image(img, heatmap)

        axes[idx].imshow(overlay)
        axes[idx].set_title(f'Condition: {conditions[label]}, Predicted: {conditions[pred]}')
        axes[idx].axis('off')
    
    # Hide empty subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
