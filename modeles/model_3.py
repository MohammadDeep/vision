import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from vision.Config import dir_dataset_folder_val, dir_dataset_folder
'''
==========================================================================
                                    path 1
==========================================================================
'''

dir_folber_val = Path(dir_dataset_folder_val , 'inter folder dataset ')
dir_folber = Path(dir_dataset_folder , 'inter folder dataset ')
from vision.Create_Dataset import Create_dataset
from vision.Config import dic_dir_val,dic_dir,dir_dataset_folder,dir_dataset_folder_val, Create_train_data, Create_val_data
from pathlib import Path

move_or_copy_files = 'copy'
if Create_train_data:
    id_class = 1
    dic_name_folber_clases = {

        f'class_id{id_class}':['core_class' ,'cut_image' , 'box_image'],
        'calss_other':['non_core_class', 'bake_image_Not']

        }

    dir_folber = Path(dir_dataset_folder , 'extended_dataset_all_folderes')

    Create_dataset (dic_name = dic_name_folber_clases ,
                         dic_dir = dic_dir,
                        dir_extended = dir_folber,
                        move_or_copy_files = move_or_copy_files
                        )


if Create_val_data:
    id_class = 1
    dic_name_folber_clases_val = {

        f'class_id{id_class}':['core_class' ,'cut_image' , 'box_image'],
        'calss_other':['non_core_class', 'bake_image_Not']

        }

    dir_folber_val = Path(dir_dataset_folder_val , 'extended_dataset_all_folderes')

    Create_dataset (dic_name = dic_name_folber_clases_val ,
                        dic_dir = dic_dir_val,
                        dir_extended = dir_folber_val,
                        move_or_copy_files = move_or_copy_files
                        )

    dic_name_folber_clases_val = {

        f'class_id{id_class}':['core_class' ],
        'calss_other':[ 'bake_image_Not']

        }

    dir_folber_val_1 = Path(dir_dataset_folder_val , 'extended_dataset_bakeImageNot_and_coreClass_folderes')

    Create_dataset (dic_name = dic_name_folber_clases_val ,
                        dic_dir = dic_dir_val,
                        dir_extended = dir_folber_val_1,
                        move_or_copy_files = move_or_copy_files
                        )


'''
==========================================================================
                                    path 2
==========================================================================
'''

# 4. آماده‌سازی داده‌ها
transform = transforms.Compose([
    transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize((224, 224)),   # تغییر اندازه به ورودی مدل
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


transform_val = transforms.Compose([
    transforms.Resize((224, 224)),   # تغییر اندازه به ورودی مدل
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

BATCH_SIZE = 2 ** 7
# ImageFolder دو کلاس را 0 و 1 نگاشت می‌کند
dataset_val = datasets.ImageFolder(root=dir_folber_val, transform=transform_val)
dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

print(dataset_val.class_to_idx)

# e.g. {'negative': 0, 'positive': 1}
# ImageFolder دو کلاس را 0 و 1 نگاشت می‌کند
dataset_train = datasets.ImageFolder(root=dir_folber, transform=transform)
dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

print(dataset_train.class_to_idx)

'''
==========================================================================
                                    path 3
==========================================================================
'''

import torch
from torch import nn
from torchvision import models

class HumanPresenceSqueezeNet(nn.Module):
    def __init__(self,
                 pretrained=True,
                 weights=models.SqueezeNet1_1_Weights.DEFAULT,
                 freeze_features_up_to=10):
        super().__init__()
        # 1) بارگذاری backbone با API جدید weights
        base = models.squeezenet1_1(pretrained =pretrained)

        # 2) ساخت classifier جدید شامل Dropout → Conv2d → Conv2d → Flatten
        #    خروجی نهایی پس از Flatten می‌شود (B, 1)
        self.classifier = nn.Sequential(
            base.classifier[0],                             # Dropout
            nn.Conv2d(512, 1, kernel_size=1, stride=1),     # جایگزینی لایه
            nn.Conv2d(1, 1, kernel_size=13, stride=1),      # همانی که قبل داشتید
            nn.Flatten()                                     # <<< این‌جا flatten!
        )

        # 3) فریز کردن لایه‌های ابتدایی features
        self.features = base.features
        for p in list(self.features.parameters())[:freeze_features_up_to]:
            p.requires_grad = False

    def forward(self, x):
        x = self.features(x)      # (B,512,13,13)
        x = self.classifier(x)    # (B,1) پس از Flatten
        return x


# مثال و проверка summary
model_3 = HumanPresenceSqueezeNet().to('cuda')

summary(model_3, (3, 224, 224))



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

'''
# 3. تعریف تابع از دست دادن (Loss) و بهینه‌ساز
# استفاده از BCEWithLogitsLoss که برای خروجی باینری سیگموید استفاده می‌شود
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pos_weight = torch.tensor([1]).to(device)
loss_function_3 = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # این تابع از سیگموید به‌طور داخلی استفاده می‌کند
optimizer_3= optim.Adam(model_3.parameters(), lr=0.0001)'''




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