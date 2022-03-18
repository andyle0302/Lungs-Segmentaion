# https://github.com/qubvel/segmentation_models.pytorch
# https://github.com/IlliaOvcharenko/lung-segmentation
# https://www.kaggle.com/pezhmansamadi/lung-segmentation-torch

import segmentation_models_pytorch as smp

import numpy as np
import torch
import time
# from sklearn.model_selection import KFold

#loss
import torch.nn as nn
import torch.nn.functional as F

# metrics
from sklearn.metrics import accuracy_score, jaccard_score, f1_score, recall_score, precision_score
from operator import add
import sys

UNet_ResNet = smp.Unet(
    encoder_name='resnet152',
    encoder_weights='imagenet', # pre_training on ImageNet
    in_channels=1, 
    classes=1
)

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class ComboLoss(nn.Module): #Dice + BCE + focal
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()

    def forward(self, inputs, targets, alpha = 0.8, gamma = 2,smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha*(1-BCE_EXP)**gamma*BCE

        Dice_BCE = BCE + dice_loss +focal_loss

        return Dice_BCE

def calculate_metrics(y_pred, y_true):
    """ Ground truth """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    y_pred = y_pred.cpu().detach().numpy()
    y_pred = y_pred > 0.5 # True False False True 
    y_pred = y_pred.astype(np.uint8) # 0 1 1 0
    y_pred = y_pred.reshape(-1) # flatten

    jaccard = jaccard_score(y_true, y_pred)
    # f1 = f1_score(y_true, y_pred)
    # recall = recall_score(y_true, y_pred)
    # precision = precision_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    # return [jaccard, f1, recall, precision, acc]
    return [jaccard, acc]


def save_checkpoint (state, filename):
    """ saving model's weights """
    print ('=> saving checkpoint')
    torch.save (state, filename)



def reset_weights(m):
    '''
    Try resetting model weights to avoid weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            # print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

def train(model, loader, optimizer, scheduler, loss_fn, metric_fn, device):
    # train(model, train_dl, optimizer, scheduler, loss_fn, metric_fn, device)
    epoch_loss = 0.0
    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    steps = len(loader)
    
    model.train()

    for i, (x, y) in enumerate (loader):
        x = x.to(device)
        # x = x.float().to(device)
        y = y.float().unsqueeze(1).to(device)

        # print('x',x)
        optimizer.zero_grad()
        y_pred = model(x) # gpu cuda

        # print('y_pred', y_pred)
        # print(y)
        # y = torch.unsqueeze(y,1)

        loss = loss_fn(y_pred, y) # comboloss BCE dice focal
        loss.backward() # ???
        
        score = metric_fn(y_pred, y)
        metrics_score = list(map(add, metrics_score, score))
        
        optimizer.step()
        learning_rate = optimizer.param_groups[0]['lr']
        

        epoch_loss += loss.item()
        
        sys.stdout.flush()
        sys.stdout.write('\r Step: [%2d/%2d], loss: %.4f - acc: %.4f' % (i, steps, loss.item(), score[1]))
    scheduler.step()
    
    # last_lr = scheduler.get_last_lr()


    sys.stdout.write('\r')
        # print('Epoch: {} \t Training Loss: {:.6f} \t Validation Loss {:.6f} \n \t ')

    epoch_loss = epoch_loss/len(loader)
    
    epoch_jaccard = metrics_score[0]/len(loader)
#     epoch_f1 = metrics_score[1]/len(loader)
    epoch_acc = metrics_score[1]/len(loader)
    
    return epoch_loss, epoch_jaccard, epoch_acc, learning_rate

def evaluate(model, loader, loss_fn, metric_fn, device):
    epoch_loss = 0.0
    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            # y = y.to(device, dtype=torch.float32)
            # x = x.float().to(device)
            y = y.float().unsqueeze(1).to(device)
            # print(x)
            # print(y)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            
            score = metric_fn(y_pred, y)
            metrics_score = list(map(add, metrics_score, score))
            
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(loader)
        
        epoch_jaccard = metrics_score[0] / len(loader)
#         epoch_f1 = metrics_score[1] / len(loader)
        epoch_acc = metrics_score[1] / len(loader)
    
    return epoch_loss, epoch_jaccard, epoch_acc    

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def fit (model, train_dl, valid_dl, optimizer, scheduler, epochs, loss_fn, metric_fn, checkpoint_path, fold, device):
    """ fiting model to dataloaders, saving best weights and showing results """
    losses, val_losses, accs, val_accs = [], [], [], []
    jaccards, val_jaccards = [], []
    learning_rate =[]

    best_val_loss = float("inf")
    patience = 8 
    checkpoint_path = checkpoint_path.replace('.pt', str(fold) + '.pt')
    since = time.time()
    for epoch in range (epochs):
        ts = time.time()
        
        loss, jaccard, acc, lr = train(model, train_dl, optimizer, scheduler, loss_fn, metric_fn, device)
        val_loss, val_jaccard, val_acc = evaluate(model, valid_dl, loss_fn, metric_fn, device)
        
        losses.append(loss)
        accs.append(acc)
        jaccards.append(jaccard)
        
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_jaccards.append(val_jaccard)

        learning_rate.append(lr)
        
        te = time.time() 

        epoch_mins, epoch_secs = epoch_time(ts, te)
        
        print ('Epoch [{}/{}], loss: {:.4f} - jaccard: {:.4f} - acc: {:.4f} - val_loss: {:.4f} - val_jaccard: {:.4f} - val_acc: {:.4f}'.format (epoch + 1, epochs, loss, jaccard, acc, val_loss, val_jaccard, val_acc))
        print(f'Time: {epoch_mins}m {epoch_secs}s')
    
        period = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(period // 60, period % 60))
        if val_loss < best_val_loss:
            count = 0
            data_str = f"===> Valid loss improved from {best_val_loss:2.4f} to {val_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print(data_str)
            best_val_loss = val_loss
            # save_checkpoint(model.state_dict(), checkpoint_path)
            torch.save(model.state_dict(), checkpoint_path) #save checkpoint
        else:
            count += 1
            # print('count = ',count)
            if count >= patience:
                print('Early stopping!')
                return dict(loss = losses, val_loss = val_losses, acc = accs, val_acc = val_accs, jaccard = jaccards, val_jaccard = val_jaccards, learning_rate = learning_rate)



    return dict(loss = losses, val_loss = val_losses, acc = accs, val_acc = val_accs, jaccard = jaccards, val_jaccard = val_jaccards, learning_rate = learning_rate)
