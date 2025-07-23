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

from torchsummary import summary


'''
==========================================================================
                                    prameters
==========================================================================
'''
print_line_len = 20
# انتخاب کن که از کدام دیتاست های استفاده میکنی
dataset_use = [
    'all_folber'
]
# محل دیتاست های اصلی که برای اموزش  این مدل استفاده شده اند
dir_folber_val = dir_dataset_orgnal_val
dir_folber = dic_dataset['all_folber']['dir']

model_name= dic_model_3['model_name']
model_stucher = dic_model_3['model_stucher']
input_shape =dic_model_3['input_shape']
mean = dic_model_3['mean']
std = dic_model_3['std']

BATCH_SIZE = 2 ** 7




'''
==========================================================================
                                    path 1
==========================================================================
'''

print('-'  * print_line_len )
print('')
for dirctory in [dir_dataset_orgnal_val, dir_dataset_orgnal]:
    dirctory = str(dirctory)
    if os.path.isdir(dirctory):
        print(f"Original datasets exist -> {dirctory}")
    else:
        raise FileNotFoundError(f"Original datasets do not exist ->{dirctory}  You must run the code -> '!python /content/vision/codes/1_Create_dataset_Folder.py'   or Or you are not on the correct path")


print('-'  * print_line_len )
print('Datasets that can be used or created.')
for key, value in dic_dataset.items():
    print('     ' ,key)
print('-'  * print_line_len )
print('The datasets used in this code.')

for name in dataset_use:
    print(f'name : {name}')
    dirctory = str(dic_dataset[name]['dir'])
    if os.path.isdir(dirctory):
        print(f"have dir {dirctory}")
    else:
        print(f"have note dir {dirctory}")
        print('Start creating the dataset')
        Create_dataet_dir(name)



'''
==========================================================================
                                    path 2
==========================================================================
'''

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
dataset_val = datasets.ImageFolder(root=dir_folber_val, transform=transform_val)
dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

print(dataset_val.class_to_idx)

# e.g. {'negative': 0, 'positive': 1}
# ImageFolder دو کلاس را 0 و 1 نگاشت می‌کند
dataset_train = datasets.ImageFolder(root=dir_folber, transform=transform)
dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

print(dataset_train.class_to_idx)

'''
==========================================================================
                                    path 3
==========================================================================
'''


# مثال و проверка summary
model_3 = model_stucher().to('cuda')

summary(model_3, (3, input_shape[0], input_shape[1]))



'''
==========================================================================
                                    path 4
==========================================================================
'''




''''
==========================================================================
                                    loss and optimizer
==========================================================================
'''


# 3. تعریف تابع از دست دادن (Loss) و بهینه‌ساز
# استفاده از BCEWithLogitsLoss که برای خروجی باینری سیگموید استفاده می‌شود
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pos_weight = torch.tensor([1]).to(device)
loss_function_3 = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # این تابع از سیگموید به‌طور داخلی استفاده می‌کند
optimizer_3= optim.Adam(model_3.parameters(), lr=0.0001)




''''
==========================================================================
                                    training
==========================================================================
'''

'''
from vision.train_val_functiones.train_val_functiones import train
model_3_results = train(model=model_3,
                        train_dataloader=dataloader_train,
                        test_dataloader=dataloader_val,
                        optimizer=optimizer_3,
                        loss_fn=loss_function_3,
                        epochs = 5
                        , number_ep = int(len(dataloader_train) // 2) + 1
                           #,results  = model_2_3_1th_results
                            )'''