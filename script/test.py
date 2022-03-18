from utils import UNet_ResNet 
from dataset import *
from operator import add

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import os
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, jaccard_score, f1_score, recall_score, precision_score
# from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

def dataloader():

    mask_path = '../dataset_lungseg/test/lungVAE-masks/' #dataset của Phú
    img_path = '../dataset_lungseg/test/images/'
    mask_list = os.listdir(mask_path)
    img_list = os.listdir(img_path)

    img_mask_list = []
    for i in range(len(mask_list)):
        m = mask_list[i].split('_mask.')
        for j in range(len(img_list)):
            img = img_list[j].rsplit('.',1)
            if m[0] == img[0] :
                img_mask_list.append((img_list[j], mask_list[i]))
    
    with open('../dataset_lungseg/test/img_mask.txt','w') as f:
        for i in range(len(img_mask_list)):
            f.write(str(img_mask_list[i])+ '\n') 

    # with open('../dataset_lungseg/test/mask.txt','w') as f:
    #     for i in range(len(mask_list)):
    #         f.write(str(mask_list[i])+ '\n') 

    transfms = A.Compose([
        A.Resize(224,224),
        A.Normalize(mean = [0.5],  std = [0.5]),
        ToTensorV2()   
    ])

    # mask_t = A.Compose([
    #     A.Resize((224,224)),
    #     A.ToTensor()
    # ])
    test_set = LungDataset(img_mask_list, img_path, mask_path, transfms)

    loader ={
        'test' : DataLoader(
            test_set, 
            batch_size=4,
            shuffle=False
        )
    }   
    # print(len(loader['test']))
    return loader

def test(dataloader, device, model, metric_fn):
   
    with torch.no_grad():
        image, y_true, y_predict = [], [], []
        metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
        model.eval()
        for x, y in dataloader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)               

            #######################
            #### METRIC###########
            #######################
            y_pred = model(x)
            score = metric_fn(y_pred, y)
            metrics_score = list(map(add, metrics_score, score))  

############################################################################

            pred = y_pred.cpu().numpy() # mask output 
            ynum = y.cpu().numpy()  # mask label

            pred = pred.reshape(len(pred), 224, 224) # 4, 224, 224
            ynum = ynum.reshape(len(ynum), 224, 224)

            pred = pred > 0.7
            pred = np.array(pred, dtype=np.uint8)

            ynum = ynum > 0.7
            ynum = np.array(ynum, dtype=np.uint8)   

            y_true.append(ynum)    
            y_predict.append(pred)

            x = x.cpu().numpy()

            x = x.reshape(len(x), 224, 224)
            x = x*0.5 + 0.5
            x = np.squeeze(x)
            x = np.clip(x, 0, 1)

            image.append(x)


        epoch_jaccard = metrics_score[0]/len(dataloader)
        epoch_f1 = metrics_score[1]/len(dataloader)
        epoch_recall = metrics_score[2]/len(dataloader)
        epoch_precision = metrics_score[3]/len(dataloader)
        epoch_acc = metrics_score[4]/len(dataloader)

    return image, y_true, y_predict, epoch_jaccard, epoch_f1, epoch_recall, epoch_precision, epoch_acc

#########################################################################################



# def predict(x, model, device): # dataset 14gb
#     model.eval()
#     with torch.no_grad():
#         x = x.to(device, dtype=torch.float32)
#         # y = y.to(device, dtype=torch.float32)

#         # output= model(x)
#         # _,pred = torch.max(output, 1)

#         y_pred = model(x)
#         pred = y_pred.cpu().numpy() # mask output 
#         # ynum = y.cpu().numpy()  # mask label

#         pred = pred.reshape(len(pred), 224, 224)
#         # ynum = ynum.reshape(len(ynum), 224, 224)

#         pred = pred > 0.5
#         pred = np.array(pred, dtype=np.uint8)

#         # ynum = ynum > 0.5
#         # ynum = np.array(ynum, dtype=np.uint8)   
#         x = x.cpu().numpy()
#         # print('len x', len(x))
#         x = x.reshape(len(x), 224, 224)
#         # print(x.shape)
#         x = x*0.5 + 0.5
#         x = np.squeeze(x)
#         # # print(input)
#         x = np.clip(x, 0, 1)

#     return x, pred
    

# image, mask = next(iter(dataloader()['test']))
# image1 = image[0][None,:,:,:]
# print(image1.shape)

# def visualize (y_true, y_pred, classes = 1):
#     cnf_matrix = confusion_matrix(y_true, y_pred)
#     fix, ax = plt.subplots(figsize = (10,10))
#     disp = ConfusionMatrixDisplay(confusion_matrix = cnf_matrix, display_labels = classes)
#     disp.plot(include_values = True, cmap = 'Blues', ax = ax, xticks_rotation = 'vertical')
#     plt.savefig('../report/FTVGG19bn/MatrixFTVGG19bn1.png')

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
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    return [jaccard, f1, recall, precision, acc]
    # return [jaccard, acc]

def main():
    jaccards, f1s, recalls, precisions, accs = [], [], [], [], []
    y = []
    cp_list = ['../model/UNet0.pt', '../model/UNet1.pt', '../model/UNet2.pt', '../model/UNet3.pt', '../model/UNet4.pt']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = UNet_ResNet.to(device)

    for fold in range(len(cp_list)):
        checkpoint = torch.load(cp_list[fold])
        model.load_state_dict(checkpoint)

        

        ############################################################################################

        x, y_true, y_pred, jaccard, f1, recall, precision, acc = test(dataloader()['test'], device, model, calculate_metrics)

        jaccards.append(jaccard)
        f1s.append(f1)
        recalls.append(recall)
        precisions.append(precision)
        accs.append(acc)        
        
        print ('Fold: {} jaccard: {:.4f} - f1: {:.4f} - recall: {:.4f} - precision: {:.4f} - acc: {:.4f}'.format (fold, jaccard, f1, recall, precision, acc))    
    # print(y)
    # imshow(x, y_true, y, '../visualize/test.png')
    # for idx in range(5):
        y.append(y_pred)
        # print(y_pred) # 
    y_avg = np.mean(y, axis=0)

    imshow(x, y_true, y, y_avg, '../visualize/test.png')

    # print(y_avg[51][0].shape)
    # plt.imshow(y_avg[51][0], cmap='gray')
    # plt.show()

    
        ############################################################################################################
        

def imshow(original,true,pred, mean, path):

    plt.figure (figsize = (15, 20))
    for idx in range(4):

        plt.subplot (4,8,idx*8 +1)
        plt.imshow(original[0][idx], cmap='gray')
        plt.title('Original Image')

        plt.subplot (4,8,idx*8 +2)
        plt.imshow(true[0][idx], cmap='gray')
        plt.title('True Mask')

        plt.subplot (4,8,idx*8 +3)
        plt.imshow(pred[0][0][idx], cmap='gray')
        plt.title('Predict Mask')

        plt.subplot (4,8,idx*8 +4)
        plt.imshow(pred[1][0][idx], cmap='gray')
        plt.title('Predict Mask')

        plt.subplot (4,8,idx*8 +5)
        plt.imshow(pred[2][0][idx], cmap='gray')
        plt.title('Predict Mask')

        plt.subplot (4,8,idx*8 +6)
        plt.imshow(pred[3][0][idx], cmap='gray')
        plt.title('Predict Mask')


        plt.subplot (4,8,idx*8 +7)
        plt.imshow(pred[4][0][idx], cmap='gray')
        plt.title('Predict Mask')

        plt.subplot (4,8,idx*8 +8)
        plt.imshow(mean[0][idx], cmap='gray')
        plt.title('Predict Mask')

        plt.savefig(path)
    plt.show()

# def main1():
#     image_tfm = A.Compose([
#         A.Resize((224, 224)),
#         A.ToTensor(),
#         A.Normalize(mean = [0.5],  std = [0.5]) 
        

#     ])

#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     model = UNet_ResNet.to(device)
    
#     checkpoint = torch.load('../model/UNet.pt')
#     model.load_state_dict(checkpoint)

#     img = Image.open('../dataset_lungseg/predict/1dad3414-88c9-4c56-af5d-3a1488af452c.png').convert('L')
#     img = image_tfm(img) 
#     img = img.unsqueeze(0)
#     # print(img.shape)

#     x, y_pred = predict(img, model, device)

#     image, mask = next(iter(dataloader()['test']))

#     # print(x.shape)

#     # print(y_pred[0][3].shape)
#     plt.figure (figsize = (15, 20))

#     plt.subplot (1,2,1)
#     plt.imshow(x, cmap='gray')
#     plt.title('Original Image')

#     plt.subplot (1,2,2)
#     plt.imshow(y_pred[0], cmap='gray')
#     plt.title('Predict Mask')

#     plt.show()

    # accuracy = accuracy_score(y_true, y_pred)
    # print(accuracy)
if __name__ == '__main__':
    main()

    


