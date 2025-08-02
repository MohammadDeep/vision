from vision.Config import dir_image_gent_val,dir_image_gent,dir_dataset_orgnal, dir_dataset_orgnal_val,Create_train_data,Create_val_data,dir_dataset_folder_val,image_dir, image_val_dir,ann_file, ann_file_val, list_calsses, dir_dataset_folder
from vision.perproses_image.Cut_Image import CutImage
from vision.Create_dase_Fimage.create_folber_dataset import getDataFoalberCoco
from pathlib import Path


import torch
import torch.nn as nn
from torchsummary import summary


train_data =  getDataFoalberCoco(
        image_dir  ,
        ann_file,
        list_calsses,
        dir_dataset_orgnal)

train_data.create_dataset_folber()


val_data =  getDataFoalberCoco(
        image_val_dir  ,
        ann_file_val,
        list_calsses,
        dir_dataset_orgnal_val)

val_data.create_dataset_folber()


print('-' * 50 )
print('Createing layer...')
class InceptionModule(nn.Module):
    def __init__(self, in_channels
                 , out_1_3,
                 out_1_5 ,
                 out_1_7
                 , out_2_3 
                 , out_2_5 
                 ,out_2_7
                 ,p_dropout = 0.5
                 ):
        super(InceptionModule, self).__init__()
        out = (out_2_3+ out_2_5 +out_2_7)
        # شاخه اول: کانولوشن 1x1
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1_3, kernel_size=1,padding='same', bias= False),
            nn.BatchNorm2d(num_features = out_1_3),
            nn.ReLU6(inplace= True),
            nn.MaxPool2d(2),
            nn.Conv2d(out_1_3 ,out_2_3, kernel_size = 3, padding='same',bias = False),
            nn.BatchNorm2d(num_features = out_2_3),

        )

        # شاخه دوم: کانولوشن 1x1 و سپس 3x3
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_1_5, kernel_size=1,padding='same', bias= False),
            nn.BatchNorm2d(num_features = out_1_5),
            nn.ReLU6(inplace= True),
            nn.MaxPool2d(2),
            nn.Conv2d(out_1_5 ,out_2_5, kernel_size = 5, padding='same',bias = False),
            nn.BatchNorm2d(num_features = out_2_5),

        )

        # شاخه سوم: کانولوشن 1x1 و سپس 5x5
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_1_7, kernel_size=1,padding='same', bias= False),
            nn.BatchNorm2d(num_features = out_1_7),
            nn.ReLU6(inplace= True),
            nn.MaxPool2d(2),
            nn.Conv2d(out_1_7 ,out_2_7, kernel_size = 7, padding='same',bias = False),
            nn.BatchNorm2d(num_features = out_2_7),

        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out, kernel_size=1,stride=2, bias= False)
        )

        self.relu = nn.ReLU6(inplace= True)
        self.dropout = nn.Dropout2d(p = p_dropout)
    def forward(self, x):
        # محاسبه خروجی هر شاخه
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out4], 1) + torch.cat([out1, out2, out3], 1)
        out = self.relu(out)
        out = self.dropout(out)
        # اتصال خروجی‌ها در امتداد بعد کانال (dim=1)
        return out

# --- مثال استفاده ---
# تعریف پارامترهای ماژول
in_channels = 192
out_1x1 = 64
red_3x3 = 96
out_3x3 = 128
red_5x5 = 16
out_5x5 = 32
out_pool = 32

# ساخت یک نمونه از ماژول
inception_layer = InceptionModule(in_channels,64, 32 , 16 , 64, 32, 16 )

# ایجاد یک تنسور ورودی تصادفی
# (batch_size, channels, height, width)
input_tensor = torch.randn(32, 192, 28, 28)

# گرفتن خروجی
output_tensor = inception_layer(input_tensor)
print ('-'  *50)
# ۱. دستگاه را به صورت داینامیک تعریف کنید
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device use : {device}')
# ۲. مدل را به دستگاه مورد نظر منتقل کنید
inception_layer.to(device)
# چاپ ابعاد خروجی
# 64 + 128 + 32 + 32 = 256

summary(inception_layer, input_tensor.shape[1:])


print('-' * 50)
print('Createing model...')
class Model_4(nn.Module):
    def __init__(self,classes_number = 1,  in_channels = 3):
        super(Model_4, self).__init__()

        self.layer_1 = InceptionModule(
            in_channels = in_channels
            ,out_1_3 = 2
            ,out_2_3 = 16
            ,out_1_5 = 2
            ,out_2_5 = 16
            ,out_1_7 = 2
            ,out_2_7 = 16
            ,p_dropout = 0.5
        )
        self.layer_2 = InceptionModule(
            in_channels = 16 * 3
            ,out_1_3 = 16
            ,out_2_3 = 32
            ,out_1_5 = 16
            ,out_2_5 = 32
            ,out_1_7 = 16
            ,out_2_7 = 32,
            p_dropout = .40
        )
        self.layer_3 = InceptionModule(
            in_channels = 32 * 3
            ,out_1_3 = 32
            ,out_2_3 = 64
            ,out_1_5 = 32
            ,out_2_5 = 64
            ,out_1_7 = 16
            ,out_2_7 = 32,
            p_dropout = .3
        )

        self.layer_4 = InceptionModule(
            in_channels = 32  + 64 * 2
            ,out_1_3 = 32
            ,out_2_3 = 128
            ,out_1_5 = 32
            ,out_2_5 = 128
            ,out_1_7 = 16
            ,out_2_7 = 32,
            p_dropout = .20
        )

        self.layer_5 = InceptionModule(
            in_channels = 32  + 128 * 2
            ,out_1_3 = 64
            ,out_2_3 = 256
            ,out_1_5 = 64
            ,out_2_5 = 256
            ,out_1_7 = 16
            ,out_2_7 = 32,
            p_dropout = .10
        )
        self.layer_6 = nn.Sequential(
            nn.AvgPool2d(7),
            nn.Flatten(),
            nn.Linear(256* 2 + 32, classes_number)
        )


    def forward(self, x):
        # محاسبه خروجی هر شاخه
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        # اتصال خروجی‌ها در امتداد بعد کانال (dim=1)
        return x

model = Model_4()
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
from vision.modeles import dic_model_3


model_name= dic_model_3['model_name']
model_stucher = dic_model_3['model_stucher']
input_shape =dic_model_3['input_shape']
mean = dic_model_3['mean']
std = dic_model_3['std']
BATCH_SIZE = 2 ** 8
'''
==========================================================================
                                    path 2
==========================================================================
'''
print('-' * 50)
print('transformes data and ceate dataset and datalober')
num_workers = int(input('enter num_workers : '))
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
dataset_train = datasets.ImageFolder(root=dir_dataset_orgnal, transform=transform)
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
                model_name = 'Model_4',
                loss_fn = loss_fn,
                results = None,
                epochs = n_epoch,
                number_ep = len(dataloader_train),
                use_sigmoid = True,
                dir_history_model = dir_history_model,
                latest_epoch = -1,
                strict = False)