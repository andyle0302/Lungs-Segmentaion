
#  https://drive.google.com/file/d/1ffbbyoPf-I3Y0iGbBahXpWqYdGd7xxQQ/view
from PIL import Image 
from torch.utils.data import Dataset
import numpy as np
import torch 

class LungDataset(Dataset):
    def __init__(self, img_mask_list, img_folder, mask_folder, transform = None): # 'Initialization'
        self.img_mask_list = img_mask_list
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.transform = transform
  
    def __len__(self):  # 'Denotes the total number of samples'
        return len(self.img_mask_list)

    def __getitem__(self,index): # 'Generates one sample of data'      

        images_names, masks_names = self.img_mask_list[index]
        images = Image.open(self.img_folder +  images_names).convert('L')# grey # kiểu PIL images
        masks = Image.open(self.mask_folder + masks_names).convert('L') # binary kiểu PIL images

        images = np.array(images, dtype=np.float32) # đổi qua numpy array kiểu float 32
        masks = np.array(masks, dtype=np.float32) # 
        masks[masks == 255] = 1 # nếu giá trị pixcel == 255 thì đưa về 1
        
        if self.transform != None:
            aug = self.transform(image = images, mask = masks)
            images = aug['image']
            masks = aug['mask']
        # masks = masks.long()
        # masks = torch.unsqueeze(masks,0)
        # print(masks.shape)
        return images, masks # chua 1 cap


