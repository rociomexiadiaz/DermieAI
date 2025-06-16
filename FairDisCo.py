import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
import os


### SETTING SEED AND DEVICE ### 

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### MODEL + LOSSES ###

class Network(torch.nn.Module):
    def __init__(self, output_size=[1,6], weights='IMAGENET1K_V1'): 
        '''
        output_size: list first is skin type, second is condition
        '''
        super(Network, self).__init__()
        bottle_neck = 256

        self.feature_extractor = models.resnet152(weights=weights)
        num_ftrs = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(num_ftrs, bottle_neck)
        # for contrastive loss
        self.project_head = nn.Sequential(
                nn.Linear(bottle_neck, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 128),
        )
        # self.activation = torch.nn.ReLU()
        # branch 1
        self.branch_1 = nn.Linear(bottle_neck, output_size[0])
        # branch 2
        self.branch_2 = nn.Linear(bottle_neck, output_size[1])
    

    def forward(self, x):
        feature_map = self.feature_extractor(x)  # (bs, bottle_neck)
        out_1 = self.branch_1(feature_map)
        out_2 = self.branch_2(feature_map)
        out_4 = self.project_head(feature_map)
        # detach feature map and pass though branch 2 again
        feature_map_detach = feature_map.detach()
        out_3 = self.branch_2(feature_map_detach)
        return [out_1, out_2, out_3, out_4]


class Confusion_Loss(torch.nn.Module):
    '''
    Confusion loss built based on the paper 'Invesgating bias and fairness.....' 
    (https://www.repository.cam.ac.uk/bitstream/handle/1810/309834/XuEtAl-ECCV2020W.pdf?sequence=1&isAllowed=y)
    '''
    def __init__(self):
        super(Confusion_Loss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, output, label):
        # output (bs, out_size). label (bs)
        prediction = self.softmax(output) # (bs, out_size)
        log_prediction = torch.log(prediction)
        loss = -torch.mean(torch.mean(log_prediction, dim=1), dim=0)

        # loss = torch.mean(torch.mean(prediction*log_prediction, dim=1), dim=0)
        return loss


class Supervised_Contrastive_Loss(torch.nn.Module):
    '''
    from https://github.com/GuillaumeErhard/Supervised_contrastive_loss_pytorch/blob/main/loss/spc.py
    https://blog.csdn.net/wf19971210/article/details/116715880
    Treat samples in the same labels as the positive samples, others as negative samples
    '''
    def __init__(self, temperature=0.1, device='cpu'):
        super(Supervised_Contrastive_Loss, self).__init__()
        self.temperature = temperature
        self.device = device
    
    def forward(self, projections, targets, attribute=None):
        # projections (bs, dim), targets (bs)
        # similarity matrix/T
        batch_size = projections.shape[0]
        dot_product_tempered = F.cosine_similarity(projections.unsqueeze(1), projections.unsqueeze(0),dim=2)/self.temperature
        # print(dot_product_tempered)
        exp_dot_tempered = torch.exp(dot_product_tempered- torch.max(dot_product_tempered, dim=1, keepdim=True)[0])+ 1e-5
        # a matrix, same labels are true, others are false
        targets = torch.argmax(targets, dim=1)
        mask_similar_class = (targets.unsqueeze(1) == targets.unsqueeze(0)).to(self.device)
        # a matrix, diagonal are zeros, others are ones
        mask_anchor_out = (1 - torch.eye(batch_size, device=self.device))
        mask_nonsimilar_class = ~mask_similar_class
        # a matrix, same labels are 1, others are 0, and diagonal are zeros
        mask_combined = mask_similar_class * mask_anchor_out
        # num of similar samples for sample
        cardinality_per_samples = torch.sum(mask_combined, dim=1)
        # print(exp_dot_tempered * mask_nonsimilar_class* mask_similar_attr)
        # print(torch.sum(exp_dot_tempered * mask_nonsimilar_class* mask_similar_attr, dim=1, keepdim=True)+exp_dot_tempered)
        if attribute != None:
            mask_similar_attr = (attribute.unsqueeze(1).repeat(1, attribute.shape[0]) == attribute).to(self.device)
            log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_nonsimilar_class * mask_similar_attr, dim=1, keepdim=True)+exp_dot_tempered+1e-5))
       
        else:
            log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_nonsimilar_class, dim=1, keepdim=True)+exp_dot_tempered+1e-5))
        supervised_contrastive_loss = torch.sum(log_prob * mask_combined)/(torch.sum(cardinality_per_samples)+1e-5)

        
        return supervised_contrastive_loss
    

### TRAINING FUNCTIONS ### 

def train_epoch(model, dataloader, device,
                criterion, optimizer, scheduler, alpha=1.0, beta=0.8):
    

    running_loss = 0.0
    model.train()
    
    total_size = 0

    for batch in dataloader:
        inputs = batch['image'].to(device)
      
        label_c, label_t = batch['diagnosis'], (batch['fst']-1)
        label_c, label_t = torch.from_numpy(np.asarray(label_c)).to(device), torch.from_numpy(np.asarray(label_t)).to(device)

        optimizer.zero_grad()

        output = model(inputs)

        loss0 = criterion[0](output[0], label_t) 
        loss1 = criterion[1](output[1], label_c)  # branch 2 confusion loss
        loss2 = criterion[2](output[2], label_c)  # branch 2 ce loss
        loss3 = criterion[3](output[3], label_t)  # supervised contrastive loss
        loss = loss0+loss1*alpha+loss2+loss3*beta

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        total_size += inputs.size(0)

    scheduler.step()
    epoch_loss = running_loss / total_size

    return epoch_loss


def val_epoch(model, dataloader, device, criterion, alpha=1.0, beta=0.8):
    
    running_loss = 0.0
    model.eval()

    total_size = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['image'].to(device)
            label_c, label_t = batch['diagnosis'], (batch['fst']-1)
            label_c, label_t = torch.from_numpy(np.asarray(label_c)).to(device), torch.from_numpy(np.asarray(label_t)).to(device)

            output = model(inputs)

            loss0 = criterion[0](output[0], label_t) 
            loss1 = criterion[1](output[1], label_c)  # branch 2 confusion loss
            loss2 = criterion[2](output[2], label_c)  # branch 2 ce loss
            loss3 = criterion[3](output[3], label_t)  # supervised contrastive loss
            loss = loss0+loss1+loss2+loss3

            running_loss += loss.item() * inputs.size(0)
            total_size += inputs.size(0)

    epoch_loss = running_loss / total_size

    return epoch_loss


def train_model(model, train_dataloader, val_dataloader, device, num_epochs=10,
                optimizer=None, scheduler=None, criterion=None, alpha=1.0, beta=0.8, run_folder=None):
    

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

    model.to(device)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    if criterion is None:
        criterion = [
            nn.MSELoss(),
            Confusion_Loss(),
            nn.CrossEntropyLoss(),
            Supervised_Contrastive_Loss(device=device)
        ]

    best_loss = float('inf')
    best_model_state = None

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_dataloader, device, criterion, optimizer, scheduler, alpha, beta)
        val_loss = val_epoch(model, val_dataloader, device, criterion, alpha, beta)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict()
            chkp_pth = os.path.join(timestamped_folder, f"best_model_epoch_{epoch+1}.pt")
            torch.save(best_model_state, chkp_pth)


    model.load_state_dict(best_model_state)

    final_chkp_pth = os.path.join(timestamped_folder, "final_model.pt")
    torch.save(best_model_state, final_chkp_pth)

    return model