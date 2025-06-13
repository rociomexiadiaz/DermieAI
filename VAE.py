from torchvision import models
import torch.nn.functional as F
from torch import nn
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

#ResNet101!

class Decoder(nn.Module):
    def __init__(self, hidden_dims=[2048,1024,512,256,128,64]):
        super(Decoder, self).__init__()

        bottle_neck = 256
        self.linear = nn.Linear(bottle_neck, 7*7*2048)

        self.layers = nn.ModuleList([])

        for i in range(len(hidden_dims)-1):
            self.layers.append(nn.Conv2d(hidden_dims[i], hidden_dims[i+1], kernel_size=(3,3), stride=1, padding=1))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm2d(hidden_dims[i+1]))
            self.layers.append(nn.Upsample(scale_factor=2))

        self.layers.append(nn.Conv2d(hidden_dims[-1], 3, kernel_size=(3,3), stride=1, padding=1))
                
    def forward(self, x):

        x = self.linear(x)
        x = x.view(x.size(0), 2048, 7, 7) 
        for layer in self.layers:
            x = layer(x)
        return x


class VAEmodel(nn.Module):

    def __init__(self, encoder=models.resnet101(weights="IMAGENET1K_V2"), num_classes=6):
        super(VAEmodel, self).__init__()
        
    
        self.num_classes = num_classes
        bottle_neck = 256
        hidden_dim = 256

        self.encoder = encoder
        num_ftrs = self.encoder.fc.in_features
        self.encoder.fc = nn.Linear(num_ftrs, bottle_neck)

        self.classifier = nn.Linear(bottle_neck, self.num_classes)
        self.latent_mu = nn.Linear(bottle_neck, hidden_dim)
        self.latent_logvar = nn.Linear(bottle_neck, hidden_dim)

        self.decoder = Decoder()

    def encode(self, x):

        h = self.encoder(x)
        mu = self.latent_mu(h)
        logvar = self.latent_logvar(h)
        preds = self.classifier(h)

        return mu, logvar, preds
    
    def decode(self, x):
        return self.decoder(x)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for sampling from latent distribution"""
        gen = torch.Generator(device='cpu')

        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.shape, generator=gen, dtype=std.dtype).to(mu.device)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar, pred = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, pred, z


class AdaptiveResampler:

    """
    Adaptive resampling based on learned latent distribution
    to address bias in training data
    """
    def __init__(self, latent_dim=256, alpha=0.01, num_bins=50):
        self.latent_dim = latent_dim
        self.alpha = alpha  # Debiasing parameter
        self.num_bins = num_bins
        self.histograms = [np.zeros(num_bins) for _ in range(latent_dim)]
        self.bin_edges = [np.linspace(-3, 3, num_bins + 1) for _ in range(latent_dim)]
        
    def update_histograms(self, latent_vars):
        """Update histograms for each latent dimension"""
        latent_vars = latent_vars.detach().cpu().numpy()
        
        for i in range(self.latent_dim):
            hist, _ = np.histogram(latent_vars[:, i], bins=self.bin_edges[i])
            # Exponential moving average for histogram update
            self.histograms[i] = 0.9 * self.histograms[i] + 0.1 * hist
    
    def compute_weights(self, latent_vars):
        """Compute sampling weights based on latent variable rarity"""
        latent_vars = latent_vars.detach().cpu().numpy()
        weights = np.ones(len(latent_vars))
        
        for i, z in enumerate(latent_vars):
            weight = 1.0
            for j in range(self.latent_dim):
                # Find which bin this latent variable falls into
                bin_idx = np.digitize(z[j], self.bin_edges[j]) - 1
                bin_idx = np.clip(bin_idx, 0, self.num_bins - 1)
                
                # Compute weight based on inverse frequency
                freq = self.histograms[j][bin_idx] + self.alpha
                weight *= 1.0 / freq
            
            weights[i] = weight
            
        # Normalize weights
        weights = weights / np.sum(weights) * len(weights)
        return weights
    

def vae_loss(x_recon, x, mu, logvar, y_pred, y_true, c1=1.0, c2=1.0, c3=0.1):
    """
    Combined loss function for DB-VAE:
    - Classification loss (cross-entropy)
    - Reconstruction loss (MSE)
    - KL divergence loss
    """
    # Classification loss
    class_criterion = nn.BCEWithLogitsLoss()
    classification_loss = class_criterion(y_pred, y_true)
    
    # Reconstruction loss
    reconstruction_loss = F.mse_loss(x_recon, x, reduction='mean')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    
    # Combined loss
    total_loss = c1 * classification_loss + c2 * reconstruction_loss + c3 * kl_loss
    
    return total_loss, classification_loss, reconstruction_loss, kl_loss


def train_epoch(model, loader, optimizer, scheduler, resampler, device):

    model.to(device)
    model.train()

    latent_variables = []

    with torch.no_grad():

        for batch in loader:
            data = batch['image'].to(device)
            target = batch['diagnosis'].to(device)

            _, mu, _, _, _ = model(data)
            latent_variables.append(mu)

        latent_variables = torch.cat(latent_variables, dim=0)
        resampler.update_histograms(latent_variables)

        weights = resampler.compute_weights(latent_variables)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        weighted_loader = DataLoader(loader.dataset, 
                                   batch_size=loader.batch_size,
                                   sampler=sampler)
        
    epoch_loss = 0
    epoch_class_loss = 0
    epoch_recon_loss = 0
    epoch_kl_loss = 0

    for batch in weighted_loader:
        data = data.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        
        x_recon, mu, logvar, y_pred, z = model(data)
        
        loss, class_loss, recon_loss, kl_loss = vae_loss(
            x_recon, data, mu, logvar, y_pred, target
        )
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_class_loss += class_loss.item()
        epoch_recon_loss += recon_loss.item()
        epoch_kl_loss += kl_loss.item()
    
    if scheduler:
        scheduler.step()

    return epoch_loss, epoch_class_loss, epoch_recon_loss, epoch_kl_loss

def val_epoch(model, loader, device):

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in loader:
            data = batch['image']
            target = batch['diagnosis']
            data = data.to(device)
            target = target.to(device)
            
            x_recon, mu, logvar, y_pred, z = model(data)
            loss, _, _, _ = vae_loss(x_recon, data, mu, logvar, y_pred, target)
            
            val_loss += loss.item()
            _, predicted = torch.max(y_pred.data, 1)
          
    val_loss = val_loss / len(loader)

    return val_loss


def train_VAE(model, train_loader, val_loader, optimizer, scheduler, resampler, num_epochs, device):


    model.to(device)

    all_train_losses = []
    all_train_class_losses = []
    all_train_recon_losses = []
    all_train_kl_losses = []

    all_val_losses = []

    best_loss = float('inf')
    best_model_state = None


    for epoch in range(num_epochs):

        train_loss, train_class_loss, train_recon_loss, train_kl_loss = train_epoch(model, train_loader, optimizer, scheduler, resampler, device)
        val_loss = val_epoch(model, val_loader, device)

        all_train_losses.append(train_loss)
        all_train_class_losses.append(train_class_loss)
        all_train_recon_losses.append(train_recon_loss)
        all_train_kl_losses.append(train_kl_loss)
        all_val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict()

    model.load_state_dict(best_model_state)

    return model







