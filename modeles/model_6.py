from vision.Config import dir_image_gent_val,dir_image_gent,dir_dataset_orgnal, dir_dataset_orgnal_val,Create_train_data,Create_val_data,dir_dataset_folder_val,image_dir, image_val_dir,ann_file, ann_file_val, list_calsses, dir_dataset_folder
from vision.perproses_image.Cut_Image import CutImage
from vision.Create_dase_Fimage.create_folber_dataset import getDataFoalberCoco
from vision.modeles import Model_6


from pathlib import Path
import torch
import torch.nn as nn
from torchsummary import summary




'''
======================================================================
                             pyper parnmaetres
======================================================================
'''

num_workers = 16 

print('-' * 50 )
print('Load model 6 ...')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model_6()
model.to(device)
print('-' * 50)
print('summary model :')
summary(model, (3, 224, 224))



import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import os
import torch
from torchvision import models
from vision.Config import dic_dataset,dir_dataset_orgnal_val, dir_dataset_orgnal
from vision.Create_Dataset import Create_dataet_dir
from vision.modeles import dic_model_6


model_name= dic_model_6['model_name']
model_stucher = dic_model_6['model_stucher']
input_shape =dic_model_6['input_shape']
mean = dic_model_6['mean']
std = dic_model_6['std']
BATCH_SIZE = 2 ** 8
'''
==========================================================================
                                    path 2
==========================================================================
'''
print('-' * 50)
print('transformes data and ceate dataset and datalober')
#num_workers = int(input('enter num_workers : '))
# 4. آماده‌سازی داده‌ها
transform = transforms.Compose([
    transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize(input_shape),   # تغییر اندازه به ورودی مدل
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


transform_val = transforms.Compose([
    transforms.Resize(input_shape),   # تغییر اندازه به ورودی مدل
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


# ImageFolder دو کلاس را 0 و 1 نگاشت می‌کند
dataset_val = datasets.ImageFolder(root=dir_dataset_orgnal_val, transform=transform_val)
dataloader_val = DataLoader(
    dataset_val, batch_size=BATCH_SIZE, shuffle=True,
      num_workers=num_workers, pin_memory=True)

print(dataset_val.class_to_idx)

# e.g. {'negative': 0, 'positive': 1}
# ImageFolder دو کلاس را 0 و 1 نگاشت می‌کند
from vision.Config import dir_extended_all_folber
dataset_train = datasets.ImageFolder(root=dir_extended_all_folber, transform=transform)
dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, 
                              shuffle=True, num_workers=num_workers
                              , pin_memory=True)

print(dataset_train.class_to_idx)


print('-' * 50)
lr =float( input('enter lr (.001):'))
  # 3. تعریف تابع از دست دادن (Loss) و بهینه‌ساز
# استفاده از BCEWithLogitsLoss که برای خروجی باینری سیگموید استفاده می‌شود
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pos_weight = torch.tensor([1]).to(device)
loss_fn= nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # این تابع از سیگموید به‌طور داخلی استفاده می‌کند
optimizer= optim.Adam(model.parameters(), lr=lr)

from vision.train_val_functiones.train_val_functiones import train
from vision.Config import dir_history_model


n_epoch = 50
print(f'start training model epoches {n_epoch}')

history = train(model,
                train_dataloader = dataloader_train,
                test_dataloader = dataloader_val,
                optimizer = optimizer,
                model_name = 'Model_4_add_dropout_3',
                loss_fn = loss_fn,
                results = None,
                epochs = n_epoch,
                number_ep = len(dataloader_train),
                use_sigmoid = True,
                dir_history_model = dir_history_model,
                latest_epoch = -1,
                strict = False)