import torch
from torch import nn
from torchvision import models
from torch.autograd import Variable
import tqdm
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt

### SETTING SEED AND DEVICE ### 

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### MODELS + LOSSES ###

# Gradient Reversal Layer (GRL) 
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * 0.1

def grad_reverse(x):
    return GradReverse.apply(x)


# Feature extractor using ResNet101
class FeatureExtractor(nn.Module):
    def __init__(self, enet=models.resnet101(weights="IMAGENET1K_V2")):
        super(FeatureExtractor, self).__init__()
       
        self.enet = enet
        self.in_ch = self.enet.fc.in_features
        self.enet.fc = nn.Identity()
        
    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x):
        feat_out = self.extract(x).squeeze(-1).squeeze(-1)
        return feat_out
    

# Classification head 
class ClassificationHead(nn.Module):
    def __init__(self, out_dim, in_ch=1536):
        super(ClassificationHead, self).__init__()
        self.layer = nn.Linear(in_ch, out_dim)
        self.activation = nn.Softmax(dim=1)  
        self.dropout = nn.Dropout(0.5)

    def forward(self, feat_out):
        x = self.layer(self.dropout(feat_out))  
       
        return x


# Auxiliary head
class AuxiliaryHead(nn.Module):
    def __init__(self, num_aux, in_ch=1536):
        super(AuxiliaryHead, self).__init__()
        self.layer = nn.Linear(in_ch, num_aux)
        self.activation = nn.Softmax(dim=1)  

    def forward(self, x_aux):
        x_aux = self.layer(x_aux)
        px_aux = self.activation(x_aux)
        return x_aux, px_aux


# Weighted loss
def criterion_func(df):
    lst = df['fitzpatrick'].value_counts().sort_index().tolist()
    lst2 = df['fitzpatrick'].value_counts().sort_index().tolist()  # placeholder

    sum_lst = sum(lst)
    sum_lst2 = sum(lst2)
    class_freq = []
    class_freq2 = []
    for i in lst:
        class_freq.append(i / sum_lst * 100)
    weights = torch.tensor(class_freq, dtype=torch.float32)
    for i in lst2:
        class_freq2.append(i / sum_lst2 * 100)
    weights2 = torch.tensor(class_freq2, dtype=torch.float32)

    weights = weights / weights.sum()
    weights2 = weights2 / weights2.sum()
    weights = 1.0 / weights
    weights2 = 1.0 / weights2
    weights = weights / weights.sum()
    weights2 = weights2 / weights2.sum()

    weights = weights.to(device)
    weights2 = weights2.to(device)
    # Note CrossEntropyLoss & BCEWithLogitsLoss includes the Softmax function so logits should be passed in (no softmax layer in model)
    criterion = nn.CrossEntropyLoss()
    criterion_aux = nn.CrossEntropyLoss(weight=weights)
    criterion_aux2 = nn.CrossEntropyLoss(weight=weights2)

    return criterion, criterion_aux, criterion_aux2


### TRAINING FUNCTIONS ###

def train_epoch_TABE(model_encoder, model_classifier, model_aux, loader, optimizer, optimizer_aux,
                     optimizer_confusion, criterion, criterion_aux, alpha, GRL=False):

    model_encoder.train()
    model_classifier.train()
    model_aux.train()

    train_loss = []
    train_loss_aux = []

    for batch in tqdm.tqdm(loader):
       
        optimizer.zero_grad()
        optimizer_aux.zero_grad()
        optimizer_confusion.zero_grad()

        data = batch['image']
        target = batch['diagnosis']
        target_aux = (batch['fst'].long() -1).view(-1) # moves FST labels from 1-6 to 0-5

        data, target, target_aux = data.to(device), target.to(device), target_aux.to(device)
        
        feat_out = model_encoder(data)  
        logits = model_classifier(feat_out)  
        #target = target.unsqueeze(1).type_as(logits)  

        # ######----------------Main Head & Pseudo Loss---------------###########

        loss_main = criterion(logits, target)  # classification loss

        _, output_conf = model_aux(feat_out)  
        
        uni_distrib = torch.FloatTensor(output_conf.size()).uniform_(0, 1)
        uni_distrib = uni_distrib.to(device)  # sending to GPU
        uni_distrib = Variable(uni_distrib)
        loss_conf = - alpha * (torch.sum(uni_distrib * torch.log(output_conf))) / float(output_conf.size(0))  # calculating confusion loss

        loss = loss_main + loss_conf  # adding main and confusion losses

        loss.backward()
        optimizer.step()
        optimizer_confusion.step()

        # ######-------------------------------Auxiliary Head Classifier Update-------------------------------###########

        optimizer.zero_grad()
        optimizer_aux.zero_grad()

        feat_out = model_encoder(data)  

        if GRL:
            feat_out = grad_reverse(feat_out)

        logits_aux, _ = model_aux(feat_out)

        loss_aux = criterion_aux(logits_aux, target_aux)
        

        loss_aux.backward()
        optimizer.step()
        optimizer_aux.step()

        loss_np = loss.detach().cpu().numpy()
        loss_aux_np = loss_aux.detach().cpu().numpy()
        train_loss.append(loss_np)
        train_loss_aux.append(loss_aux_np)

    smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
    smooth_loss_aux = sum(train_loss_aux[-100:]) / min(len(train_loss_aux), 100)
    print(f'Train Loss: {smooth_loss:.4f}, Train Loss Aux: {smooth_loss_aux:.4f}')
    return train_loss, train_loss_aux


def eval_epoch_TABE(model_encoder, model_classifier, model_aux, loader, criterion, criterion_aux, GRL=False):
    model_encoder.eval()
    model_classifier.eval()
    model_aux.eval()

    eval_loss = []
    eval_loss_aux = []

    with torch.no_grad():
        for batch in tqdm.tqdm(loader):
            data = batch['image']
            target = batch['diagnosis']
            target_aux = (batch['fst'].long() -1).view(-1)

            data, target, target_aux = data.to(device), target.to(device), target_aux.to(device)

            feat_out = model_encoder(data)  
            logits = model_classifier(feat_out)  

            loss_main = criterion(logits, target)  

            _, output_conf = model_aux(feat_out)  
            
            uni_distrib = torch.FloatTensor(output_conf.size()).uniform_(0, 1)
            uni_distrib = uni_distrib.to(device)  
            uni_distrib = Variable(uni_distrib)
            loss_conf = - torch.sum(uni_distrib * torch.log(output_conf)) / float(output_conf.size(0))  

            loss = loss_main + loss_conf  

            eval_loss.append(loss.detach().cpu().numpy())

            if GRL:
                feat_out = grad_reverse(feat_out)

            logits_aux, _ = model_aux(feat_out)

            loss_aux = criterion_aux(logits_aux, target_aux)

            eval_loss_aux.append(loss_aux.detach().cpu().numpy())

    smooth_loss = sum(eval_loss[-100:]) / min(len(eval_loss), 100)
    smooth_loss_aux = sum(eval_loss_aux[-100:]) / min(len(eval_loss_aux), 100)
    print(f'Eval Loss: {smooth_loss:.4f}, Eval Loss Aux: {smooth_loss_aux:.4f}')
    
    return eval_loss, eval_loss_aux


def train_model(model_encoder, model_classifier, model_aux, train_loader, val_loader, num_epochs, optimizer, 
                optimizer_aux,optimizer_confusion, criterion, criterion_aux, device, alpha=0.1, GRL=False, run_folder=None):
    
    # Saving the model checkpoints
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("checkpoints", exist_ok=True)

    if run_folder is None:
        timestamped_folder = os.path.join("checkpoints", f'run_{timestamp}')
        
    else:
        named_run_folder = os.path.join("checkpoints", run_folder)
        os.makedirs(named_run_folder, exist_ok=True)
        
        timestamped_folder = os.path.join(named_run_folder, f'run_{timestamp}')

    os.makedirs(timestamped_folder, exist_ok=True)
    
    model_encoder.to(device)
    model_classifier.to(device)
    model_aux.to(device)

    best_val_loss = float('inf')
    best_model_state = None

    train_losses = []
    val_losses = []
    train_losses_aux = []
    val_losses_aux = []

    model_encoder.to(device)
    model_classifier.to(device)
    model_aux.to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')

        train_loss, train_loss_aux = train_epoch_TABE(
            model_encoder, model_classifier, model_aux, train_loader, optimizer, optimizer_aux,
            optimizer_confusion, criterion, criterion_aux, alpha, GRL
        )

        train_losses.append(train_loss)
        train_losses_aux.append(train_loss_aux)

        eval_loss, eval_loss_aux = eval_epoch_TABE(
            model_encoder, model_classifier, model_aux, val_loader, criterion, criterion_aux, GRL
        )

        val_losses.append(eval_loss)
        val_losses_aux.append(eval_loss_aux)

        if np.mean(eval_loss) < best_val_loss:
            best_val_loss = np.mean(eval_loss)
            chkp_pth = os.path.join(timestamped_folder, f"best_model_epoch_{epoch+1}.pt")
            best_model_state = {
                'encoder_state_dict': model_encoder.state_dict(),
                'classifier_state_dict': model_classifier.state_dict(),
                'auxiliary_state_dict': model_aux.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optimizer_aux_state_dict': optimizer_aux.state_dict(),
                'optimizer_confusion_state_dict': optimizer_confusion.state_dict(),
            }

            torch.save(best_model_state, chkp_pth)

    final_chkp_pth = os.path.join(timestamped_folder, "final_model.pt")
    torch.save({
                'encoder_state_dict': model_encoder.state_dict(),
                'classifier_state_dict': model_classifier.state_dict(),
                'auxiliary_state_dict': model_aux.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optimizer_aux_state_dict': optimizer_aux.state_dict(),
                'optimizer_confusion_state_dict': optimizer_confusion.state_dict(),
            }, final_chkp_pth)
    
    model_encoder.load_state_dict(best_model_state['encoder_state_dict'])
    model_classifier.load_state_dict(best_model_state['classifier_state_dict'])
    model_aux.load_state_dict(best_model_state['auxiliary_state_dict'])

    # Plot losses
    epochs = range(1, num_epochs + 1)
    fig = plt.figure(figsize=(12, 8))

    plt.plot(epochs, train_losses, label='Train Total Loss')
    plt.plot(epochs, val_losses, label='Val Total Loss')
    plt.plot(epochs, train_losses_aux, label='Train Aux Loss', linestyle='--')
    plt.plot(epochs, val_losses_aux, label='Val Aux Loss', linestyle='--')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    return model_encoder, model_classifier, model_aux, fig
