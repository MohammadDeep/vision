import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import os
import torch
from torchvision import models




class HumanPresenceSqueezeNet(nn.Module):
    def __init__(self,
                 pretrained=True,
                 weights=models.SqueezeNet1_1_Weights.DEFAULT,
                 freeze_features_up_to=10):
        super().__init__()
        # 1) بارگذاری backbone با API جدید weights
        weights = models.SqueezeNet1_1_Weights.DEFAULT
        base = models.squeezenet1_1(weights=weights)

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











dic_model_2 = {
    'model_name' : 'model_2',
    'model_stucher' : HumanPresenceSqueezeNet,
    'input_shape' : (224, 224),
    'mean' : [0.485, 0.456, 0.406],
    'std' :[0.229, 0.224, 0.225],
}



dic_model_3 = {
    'model_name' : 'model_2',
    'model_stucher' : HumanPresenceSqueezeNet,
    'input_shape' : (224, 224),
    'mean' : [0.485, 0.456, 0.406],
    'std' :[0.229, 0.224, 0.225],
}