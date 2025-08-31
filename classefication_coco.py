import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
import os
import torch.nn as nn
import torch.optim as optim
from vision.Config import dir_history_model
from vision.train_val_functiones.train_val_functiones import train
from torchsummary import summary

# ----------------------------------------------------------------------
#                       Parameters
# ----------------------------------------------------------------------
n_epoch = 200
lr = 0.0001
BATCH_SIZE = 2 ** 7
NUM_WORKERS = 14
SIZE_INPUT = 256  # input size for image (256, 256)
NUM_CLASSES = 80  # Number of classes for COCO dataset

# Dataset directories
annotation_file_train = 'annotations/instances_train2017.json'
img_dir_train = 'train2017'
annotation_file_val = 'annotations/instances_val2017.json'
img_dir_val = 'val2017'

# ----------------------------------------------------------------------
#                      Create Dataset and DataLoader
# ----------------------------------------------------------------------
class CocoMultiLabelDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None, num_classes=NUM_CLASSES):
        """
        :param annotation_file: JSON file path containing annotations
        :param img_dir: Directory containing the images
        :param transform: Transformations to apply
        :param num_classes: Number of classes (for COCO, it's 80 classes)
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

        # Get annotations for the image
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
        labels = [ann['category_id'] for ann in annotations]

        # Convert labels to binary vector
        target = torch.zeros(self.num_classes)
        for label in labels:
            if label <= self.num_classes:  # Ensure the label is within bounds
                target[int(label - 1)] = 1  # Class labels start from 1, so subtract 1 to align with 0-index

        if self.transform:
            img = self.transform(img)

        return img, target.float()

# ----------------------------------------------------------------------
#                       Transforms for Data Augmentation
# ----------------------------------------------------------------------
transform = transforms.Compose([
    transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize((SIZE_INPUT, SIZE_INPUT)),  # Resize to input size for the model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((SIZE_INPUT, SIZE_INPUT)),  # Resize to input size for the model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ----------------------------------------------------------------------
#                         Create Datasets and DataLoaders
# ----------------------------------------------------------------------
# Training dataset and dataloader
train_dataset = CocoMultiLabelDataset(annotation_file=annotation_file_train, img_dir=img_dir_train, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

# Validation dataset and dataloader
val_dataset = CocoMultiLabelDataset(annotation_file=annotation_file_val, img_dir=img_dir_val, transform=transform_val)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# ----------------------------------------------------------------------
#                        Model Setup
# ----------------------------------------------------------------------
from vision.modeles import Model_8  # Make sure to import your model properly

model = Model_8(classes_number=NUM_CLASSES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss function and optimizer
pos_weight = torch.tensor([1]).to(device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # Uses sigmoid internally
optimizer = optim.Adam(model.parameters(), lr=lr)

# ----------------------------------------------------------------------
#                         Model Summary
# ----------------------------------------------------------------------
print('-' * 50)
print('Loading model...')
print('-' * 50)
print('Model summary:')
summary(model, (3, SIZE_INPUT, SIZE_INPUT))

# ----------------------------------------------------------------------
#                         Training Process
# ----------------------------------------------------------------------
history = train(model,
                train_dataloader=train_loader,
                test_dataloader=val_loader,
                optimizer=optimizer,
                model_name='model_8_80_classes',
                loss_fn=loss_fn,
                results=None,
                epochs=n_epoch,
                number_ep=len(train_loader),
                use_sigmoid=True,
                dir_history_model=dir_history_model,
                latest_epoch=-1,
                strict=False)
