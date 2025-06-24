import torch
import datetime
import os   
import matplotlib.pyplot as plt

def train_epoch(model, train_loader, optimizer, criterion, device):

    model.train()

    epoch_loss = 0

    for batch in train_loader:
        images = batch['image'].to(device)
        labels = batch['diagnosis'].to(device) 
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)

    return epoch_loss


def validate_epoch(model, val_loader, criterion, device):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['diagnosis'].to(device) 
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            epoch_loss += loss.item()

    epoch_loss /= len(val_loader)

    return epoch_loss

def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=5, scheduler=None):
    
    model.to(device)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if scheduler:
            scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            best_model_state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }

    model.load_state_dict(best_model_state['model_state_dict'])

    # Plot losses
    epochs = range(1, num_epochs + 1)
    fig = plt.figure(figsize=(12, 8))

    plt.plot(epochs, train_losses, label='Train Total Loss')
    plt.plot(epochs, val_losses, label='Val Total Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    return model, fig



