import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.optim as optim
from vision.Config import dir_history_model
from vision.train_val_functiones.train_val_functiones import train
from torchsummary import summary

'''
==========================================================================
                                PARAMETER
==========================================================================
'''
n_epoch = 200
lr = .0001
BATCH_SIZE = 2 ** 7
NUM_WORKERS = 14
SIZE_INPUT = 256 # input size for image for exampel (256, 256)


# dir dataset
annotation_file_train='annotations/instances_train2017.json'
img_dir_train='train2017'
annotation_file_val='annotations/instances_val2017.json'
img_dir_val='val2017'

'''
==========================================================================
                                Create dataset and dataloaber
==========================================================================
'''
class CocoMultiLabelDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None, num_classes=80):
        """
        :param annotation_file: مسیر فایل JSON حاوی برچسب‌ها
        :param img_dir: مسیر فولدر تصاویر
        :param transform: اعمال پیش‌پردازش‌ها
        :param num_classes: تعداد کلاس‌ها (برای COCO، تعداد کلاس‌ها 80 است)
        """
        self.coco = COCO(annotation_file)
        self.img_dir = img_dir
        self.transform = transform
        self.num_classes = num_classes
        self.ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        # دریافت لیبل‌های چندکلاسه
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
        labels = [ann['category_id'] for ann in annotations]

        # تبدیل لیبل‌ها به بردار باینری
        target = torch.zeros(self.num_classes)
        for label in labels:
            target[label - 1] = 1  # کلاس‌ها از 1 شروع می‌شوند، بنابراین باید 1 کم کنیم

        if self.transform:
            img = self.transform(img)

        return img, target.float()


transform_val = transforms.Compose([
    #transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize((SIZE_INPUT, SIZE_INPUT)),   # تغییر اندازه به ورودی مدل
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. آماده‌سازی داده‌ها
transform = transforms.Compose([
    transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize((SIZE_INPUT, SIZE_INPUT)),   # تغییر اندازه به ورودی مدل
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# بارگذاری دیتاست
train_dataset = CocoMultiLabelDataset(annotation_file= annotation_file_train
                                      , img_dir= img_dir_train
                                      , transform=transform)

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, 
                              shuffle = True, num_workers = NUM_WORKERS
                              , pin_memory = True)
val_dataset = CocoMultiLabelDataset(annotation_file= annotation_file_val
                                      , img_dir= img_dir_val
                                      , transform=transform_val)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, 
                              shuffle = False, num_workers = NUM_WORKERS
                              , pin_memory = True)



'''
==========================================================================
                                Create Model
==========================================================================
'''


from vision.modeles import Model_8



model = Model_8(classes_number = 80)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pos_weight = torch.tensor([1]).to(device)
loss_fn= nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # این تابع از سیگموید به‌طور داخلی استفاده می‌کند
optimizer= optim.Adam(model.parameters(), lr=lr)
loss = nn.BCEWithLogitsLoss()



print('-' * 50 )
print('Load model 8 ...')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model_8()
model.to(device)
print('-' * 50)
print('summary model :')
summary(model, (3, 256, 256))


history = train(model,
                train_dataloader = train_loader,
                test_dataloader = val_loader,
                optimizer = optimizer,
                model_name = 'model_8_80_calsses',
                loss_fn = loss_fn,
                results = None,
                epochs = n_epoch,
                number_ep = len(train_loader),
                use_sigmoid = True,
                dir_history_model = dir_history_model,
                latest_epoch = -1,
                strict = False)