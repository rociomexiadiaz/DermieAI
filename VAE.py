from torchvision import models
import torch.nn.functional as F
from torch import nn
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt


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


class Decoder(nn.Module):
    """
    Decoder that exactly mirrors ResNet-152 encoder
    
    ResNet-152 forward: 224×224×3 → 56×56×64 → 56×56×256 → 28×28×512 → 14×14×1024 → 7×7×2048
    This decoder:       latent_dim → 7×7×2048 → 14×14×1024 → 28×28×512 → 56×56×256 → 56×56×64 → 224×224×3
    """
    
    def __init__(self, latent_dim=120):
        super(Decoder, self).__init__()
        
        # Project latent to 7x7x2048 (start of decoder)
        self.latent_projection = nn.Linear(latent_dim, 7 * 7 * 2048)
        
        # Reverse of ResNet layers
        # Layer 4 reverse: 7x7x2048 -> 14x14x1024
        self.layer4_reverse = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        # Layer 3 reverse: 14x14x1024 -> 28x28x512
        self.layer3_reverse = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Layer 2 reverse: 28x28x512 -> 56x56x256
        self.layer2_reverse = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Layer 1 reverse: 56x56x256 -> 56x56x64 (no upsampling)
        self.layer1_reverse = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final reverse of conv1+maxpool: 56x56x64 -> 224x224x3
        self.final_decode = nn.Sequential(
            # Reverse maxpool (2x2) and conv1: 56x56x64 -> 224x224x3
            nn.ConvTranspose2d(64, 3, kernel_size=8, stride=4, padding=2),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, z):
        # Project latent vector to feature map
        x = self.latent_projection(z)  # [B, 7*7*2048]
        x = x.view(x.size(0), 2048, 7, 7)  # [B, 2048, 7, 7]
        
        # Reverse ResNet layers
        x = self.layer4_reverse(x)  # [B, 1024, 14, 14]
        x = self.layer3_reverse(x)  # [B, 512, 28, 28] 
        x = self.layer2_reverse(x)  # [B, 256, 56, 56]
        x = self.layer1_reverse(x)  # [B, 64, 56, 56]
        x = self.final_decode(x)    # [B, 3, 224, 224]
        
        return x


class VAEmodel(nn.Module):

    def __init__(self, encoder=models.resnet101(weights="IMAGENET1K_V2"), num_classes=6):
        super(VAEmodel, self).__init__()
        
    
        self.num_classes = num_classes
        bottle_neck = 256
        hidden_dim = 120

        self.encoder = encoder
        for name, param in self.encoder.named_parameters():
            if 'layer4' in name or 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
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
    def __init__(self, latent_dim=120, alpha=0.01, num_bins=50):
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
            log_weight_sum = 0

            for j in range(self.latent_dim):
                # Find which bin this latent variable falls into
                bin_idx = np.digitize(z[j], self.bin_edges[j]) - 1
                bin_idx = np.clip(bin_idx, 0, self.num_bins - 1)
                
                # Compute weight based on inverse frequency
                freq = self.histograms[j][bin_idx] + self.alpha
                log_weight_sum += -np.log(freq)

            log_weight = log_weight_sum / self.latent_dim
            
            weight = np.exp(log_weight)
            weights[i] = weight
                
        if np.sum(weights) == 0:
            print("Warning: all weights are zero. Using uniform weights.")
            weights = np.ones_like(weights)

        weights = weights / np.sum(weights) * len(weights)
        return weights

def get_beta(epoch, max_epochs, beta_start=0.0, beta_end=1.0):
    """Gradually increase beta to prevent KL collapse"""
    if epoch < max_epochs * 0.8:  
        return beta_start + (beta_end - beta_start) * (epoch / (max_epochs * 0.5))
    else:  
        return beta_end

def vae_loss(epoch, x_recon, x, mu, logvar, y_pred, y_true, c1=1.0, c2=1, c3=0.1, use_clip=False):
    """
    Combined loss function for DB-VAE:
    - Classification loss (cross-entropy)
    - Reconstruction loss (MSE)
    - KL divergence loss
    """

    current_beta = get_beta(epoch, max_epochs=20, beta_start=0.0, beta_end=c3)
                            
    # Classification loss
    class_criterion = nn.BCEWithLogitsLoss()
    classification_loss = class_criterion(y_pred, y_true)
    
    # Reconstruction loss
    if use_clip:
        reconstruction_loss = torch.tensor(0.0, device=y_pred.device)
        c2 = 0.0  # Disable reconstruction weight
    else:
        reconstruction_loss = F.mse_loss(x_recon, x, reduction='mean')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Combined loss
    total_loss = c1 * classification_loss + c2 * reconstruction_loss + c3 * kl_loss
    
    return total_loss, classification_loss, reconstruction_loss, kl_loss


def train_epoch(epoch, model, loader, optimizer, scheduler, resampler, device, use_clip=False):

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
        data = batch['image'].to(device)
        target = batch['diagnosis'].to(device)
        
        optimizer.zero_grad()
        
        x_recon, mu, logvar, y_pred, z = model(data)
        
        loss, class_loss, recon_loss, kl_loss = vae_loss(epoch,
            x_recon, data, mu, logvar, y_pred, target, use_clip=use_clip
        )
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_class_loss += class_loss.item()
        epoch_recon_loss += recon_loss.item()
        epoch_kl_loss += kl_loss.item()
    
    if scheduler:
        scheduler.step()

    epoch_loss = epoch_loss / len(weighted_loader)
    epoch_class_loss = epoch_class_loss / len(weighted_loader)
    epoch_recon_loss = epoch_recon_loss / len(weighted_loader)
    epoch_kl_loss = epoch_kl_loss / len(weighted_loader)

    return epoch_loss, epoch_class_loss, epoch_recon_loss, epoch_kl_loss

def val_epoch(epoch, model, loader, device, use_clip=False):

    model.eval()
    val_loss = 0
    val_class_loss = 0
    val_recon_loss = 0
    val_kl_loss = 0

    with torch.no_grad():
        for batch in loader:
            data = batch['image']
            target = batch['diagnosis']
            data = data.to(device)
            target = target.to(device)
            
            x_recon, mu, logvar, y_pred, z = model(data)
            loss, class_loss, recon_loss, kl_loss = vae_loss(epoch, x_recon, data, mu, logvar, y_pred, target, use_clip=use_clip)
            
            val_loss += loss.item()
            val_class_loss += class_loss.item()
            val_recon_loss += recon_loss.item()
            val_kl_loss += kl_loss.item()
            _, predicted = torch.max(y_pred.data, 1)
          
    val_loss = val_loss / len(loader)
    val_class_loss = val_class_loss / len(loader)
    val_recon_loss = val_recon_loss / len(loader)
    val_kl_loss = val_kl_loss / len(loader)

    return val_loss, val_class_loss, val_recon_loss, val_kl_loss


def train_VAE(model, train_loader, val_loader, optimizer, scheduler, resampler, num_epochs, device, use_clip=False):

    model.to(device)

    all_train_losses = []
    all_train_class_losses = []
    all_train_recon_losses = []
    all_train_kl_losses = []

    all_val_losses = []
    all_val_class_losses = []
    all_val_recon_losses = []
    all_val_kl_losses = []

    best_loss = float('inf')
    best_model_state = None


    for epoch in range(num_epochs):

        train_loss, train_class_loss, train_recon_loss, train_kl_loss = train_epoch(epoch, model, train_loader, optimizer, scheduler, resampler, device, use_clip=use_clip)
        val_loss, val_class_loss, val_recon_loss, val_kl_loss = val_epoch(epoch, model, val_loader, device, use_clip=use_clip)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        all_train_losses.append(train_loss)
        all_train_class_losses.append(train_class_loss)
        all_train_recon_losses.append(train_recon_loss)
        all_train_kl_losses.append(train_kl_loss)
        all_val_losses.append(val_loss)
        all_val_class_losses.append(val_class_loss)
        all_val_recon_losses.append(val_recon_loss)
        all_val_kl_losses.append(val_kl_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict()

    model.load_state_dict(best_model_state)

     # Plot losses
    epochs = range(1, num_epochs + 1)
    fig = plt.figure(figsize=(12, 8))

    
    plt.plot(epochs, all_train_losses, label='Train Total Loss')
    plt.plot(epochs, all_train_class_losses, label='Train Classification Loss', linestyle='--')
    plt.plot(epochs, all_train_recon_losses, label='Train Reconstruction Loss', linestyle='--')
    plt.plot(epochs, all_train_kl_losses, label='Train KL Divergence Loss', linestyle='--')
    plt.plot(epochs, all_val_losses, label='Val Total Loss')
    plt.plot(epochs, all_val_class_losses, label='Val Classification Loss', linestyle='--')
    plt.plot(epochs, all_val_recon_losses, label='Val Reconstruction Loss', linestyle='--')
    plt.plot(epochs, all_val_kl_losses, label='Val KL Divergence Loss', linestyle='--')
    plt.ylim((0,2))
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    return model, fig







