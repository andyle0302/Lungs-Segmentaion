# https://www.kaggle.com/pezhmansamadi/lung-segmentation-torch
from script.utils import *
from script.dataset import *
from script.visualize import *
from script.model import *

#importing the libraries
import os
import numpy as np
# import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
#for reading and displaying images
import matplotlib.pyplot as plt

#Pytorch libraries and modules
import torch

from torchsummary import summary

from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, ConcatDataset

#torchvision for pre-trained models and augmentation

#for evaluating model

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import argparse

from torch.nn import BCEWithLogitsLoss


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--CHECKPOINT_PATH", default = './model/UNet.pt',type=str)
    parser.add_argument('--img_path', default='./dataset_lungseg/images/', type=str)
    parser.add_argument('--mask_path', default='./dataset_lungseg/masks/', type= str)

    parser.add_argument('--BATCH_SIZE', default=8, type=int)
    parser.add_argument('--num_epochs', default= 15, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)

    opt = parser.parse_args()
    return opt



def dataloader():

    mask_path = './dataset_lungseg/masks/'
    img_path = './dataset_lungseg/images/'
    mask_list = os.listdir(mask_path)
    img_mask_list = [(mask_names.replace('_mask',''), mask_names) for mask_names in mask_list]

    train_list, val_list = train_test_split(img_mask_list, test_size = 0.2, random_state = 42) 

    # train_list, val_list = train_test_split(train_list,test_size = 0.1, random_state = 42)
    # print(len(train_list), len(val_list), len(test_list)) # 506, 57, 141
    # print(train_list)

    aug = A.Compose([
        A.Resize(256, 256), 
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.RandomGamma(),
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.9),            
        ], p = 0.3),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),         
        A.Normalize(mean = [0.5],  std = [0.5]),
        ToTensorV2()   
    ])

    transfm = A.Compose([
        A.Resize(256,256),
        A.Normalize(mean = [0.5],  std = [0.5]),
        ToTensorV2()
    ])
  
    train_set = LungDataset(train_list, img_path, mask_path, aug)
    val_set = LungDataset(val_list, img_path, mask_path, transfm)
    # test_set = LungDataset(test_list, img_path, mask_path, transform = (image_t, mask_t))

    loader ={
        'train' : DataLoader(
            train_set, 
            batch_size= 8,
            shuffle=True
        ),
        'val' : DataLoader(
            val_set, 
            batch_size=8,
            shuffle=True
        ),
        # 'test' : DataLoader(
        #     test_set, 
        #     batch_size=4,
        #     shuffle=True
        # )
    }   
    return loader

def datasetKfold(kfold):

    mask_path = './dataset_lungseg/masks/'
    img_path = './dataset_lungseg/images/'
    mask_list = os.listdir(mask_path)
    img_mask_list = [(mask_names.replace('_mask',''), mask_names) for mask_names in mask_list]
    
    idx = kfold.split(img_mask_list) 
    # dataset = LungDataset(img_mask_list, img_path, mask_path, ToTensorV2())
    return idx, img_mask_list

def get_item_to_idx(idx_list,image_list):
  train_list = [image_list[i] for i in idx_list]
  return train_list

def main():
    opt = get_opt()
    

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = UNet_ResNet.to(device)
    optimizer = Adam(model.parameters(), opt.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # loss_fn = DiceBCELoss()
    # loss_fn = BCEWithLogitsLoss()
    loss_fn = ComboLoss()

    torch.cuda.empty_cache()

    # train and validation
    res = fit(model, dataloader()['train'], dataloader()['val'], optimizer, scheduler, opt.num_epochs, loss_fn, calculate_metrics, opt.CHECKPOINT_PATH, device)
    
    # visualize loss, acc
    loss, val_loss = res['loss'], res['val_loss']
    acc, val_acc = res['acc'], res['val_acc']
    plot_acc_loss (loss, val_loss, acc, val_acc, './visualize/loss_acc')

    # test
    # with torch.no_grad():
    #     for x, y in dataloader()['val']:
    #         x = x.to(device)
    #         y = y.to(device)

    #         y_pred = model(x)
    #         break

    # pred = y_pred.cpu().numpy()
    # ynum = y.cpu().numpy()

    # pred = pred.reshape(len(pred), 224, 224)
    # ynum = ynum.reshape(len(ynum), 224, 224)

    # pred = pred > 0.5
    # pred = np.array(pred, dtype=np.uint8)

    # ynum = ynum > 0.5
    # ynum = np.array(ynum, dtype=np.uint8)

    # plt.imshow(pred[0], cmap='gray')
    # plt.imshow(ynum[0], cmap='gray')
    # plt.show()

def mainKFold():
    opt = get_opt()
    aug = A.Compose([
        A.Resize(256, 256), 
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.RandomGamma(),
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.9),            
        ], p = 0.3),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),         
        A.Normalize(mean = [0.5],  std = [0.5]),
        ToTensorV2()   
    ])

    transfm = A.Compose([
        A.Resize(256,256),
        A.Normalize(mean = [0.5],  std = [0.5]),
        ToTensorV2()
    ])

    k_folds = 5
    # For fold results
    results = {}
    # Set fixed random number seed
    torch.manual_seed(42)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    loss_fn = ComboLoss()

    torch.cuda.empty_cache()

    # train and validation

    ###########################
    ############K-FOLD#########
    ###########################
    #define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)
    idx, img_mask_list = datasetKfold(kfold)

    # print(next(iter(idx)))
    #Start print
    print('=======================================')

    for fold, (train_ids, test_ids) in enumerate(idx): #goi vay no sai

        #Print
        print(f'FOLD {fold}')
        print('-------------------------')
        train_ids = get_item_to_idx(train_ids, img_mask_list)
        test_ids  = get_item_to_idx(test_ids, img_mask_list)

        train_subsampler = LungDataset(train_ids,opt.img_path, opt.mask_path,aug)  
        test_subsampler = LungDataset(test_ids,opt.img_path, opt.mask_path,transfm)

        trainloader = DataLoader(
            train_subsampler,
            batch_size=8,
            shuffle=True
        )

        testloader = DataLoader(
            test_subsampler,
            batch_size=8,
            shuffle=True
        )

        model = UNet_ResNet.to(device)
        model.apply(reset_weights)
        optimizer = Adam(model.parameters(), opt.lr)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        scheduler3 = lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10)
        # loss_fn = DiceBCELoss()
        # loss_fn = BCEWithLogitsLoss()
        loss_fn = ComboLoss()
        
        res = fit(model, trainloader, testloader, optimizer, scheduler3, opt.num_epochs, loss_fn, calculate_metrics, opt.CHECKPOINT_PATH, fold, device)
    
        # visualize loss, acc
        loss, val_loss = res['loss'], res['val_loss']
        acc, val_acc = res['acc'], res['val_acc']

        Accuracy = 0
        for k in range(len(val_acc)):
            Accuracy +=  val_acc[k]
        
        # Print accuracy
        print ('Accuracy for fold %d: %f %%' % (fold, 100* Accuracy/len(val_acc)))
        print('--------------------------------')
        results[fold] = 100* Accuracy/len(val_acc)

      # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum/len(results.items())} %')


##################################################################################
    plot_acc_loss (loss, val_loss, acc, val_acc, './visualize/loss_acc')

    plot_LR(res['learning_rate'], './visualize/LR')

    print(res['learning_rate'])
  

if __name__ == '__main__':
    mainKFold()
    # dataloader1()
    # image, mask = next(iter(dataloader()['train']))

    # # print(x.shape)

    # # print(y_pred[0][3].shape)
    # image = image.squeeze()
    # mask = mask.squeeze()
    # plt.figure (figsize = (15, 20))

    # plt.subplot (1,2,1)
    # plt.imshow(image[0], cmap='gray')
    # plt.title('Original Image')

    # plt.subplot (1,2,2)
    # plt.imshow(mask[0], cmap='gray')
    # plt.title('True Mask')

    # plt.show()


    