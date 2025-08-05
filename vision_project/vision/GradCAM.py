import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
from torchvision.datasets import ImageFolder
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
from typing import List

# بخش ۱: کلاس GradCAM و توابع کمکی (بدون تغییر زیاد)

class GradCAM:
    """
    کلاس اصلی برای محاسبه نقشه حرارتی Grad-CAM.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_tensor: torch.Tensor, target_class_idx: int = None) -> np.ndarray:
        self.model.eval()
        output = self.model(input_tensor)

        if target_class_idx is None:
            target_class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class_idx].backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()

        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = nn.functional.relu(heatmap)
        
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)
            
        return heatmap.cpu().numpy()

def find_last_conv_layer(model: nn.Module) -> nn.Module:
    """
    آخرین لایه Conv2d را در یک مدل به صورت خودکار پیدا می‌کند.
    """
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv2d):
            print(f"لایه هدف به صورت خودکار پیدا شد: {name}")
            return module
    raise ValueError("هیچ لایه Conv2d در مدل پیدا نشد.")

def show_cam_on_image(img: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """
    نقشه حرارتی را روی تصویر اصلی ترکیب کرده و نمایش می‌دهد.
    """
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + img * 0.6
    return np.clip(superimposed_img, 0, 255).astype(np.uint8)

def tensor_to_img(tensor: torch.Tensor) -> np.ndarray:
    """
    تنسور ورودی را به یک تصویر قابل نمایش OpenCV تبدیل می‌کند.
    """
    tensor = tensor.squeeze(0)
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    img = inv_normalize(tensor)
    img = img.permute(1, 2, 0)
    img = (img.numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# بخش ۲: تابع اصلی و جدید برای تحلیل

def analyze_and_plot_indices(model: nn.Module, 
                             target_layer: nn.Module, 
                             dataset: torch.utils.data.Dataset, 
                             indices_to_plot: List[int]):
    """
    تابع اصلی که تصاویر با اندیس‌های مشخص شده را از دیتاست گرفته
    و برای آن‌ها نقشه حرارتی Grad-CAM را رسم می‌کند.

    Args:
        model: مدل PyTorch آموزش‌دیده.
        target_layer: لایه کانولوشنی هدف برای تحلیل.
        dataset: دیتاست حاوی تصاویر.
        indices_to_plot: لیستی از اندیس تصاویری که باید تحلیل شوند.
    """
    grad_cam = GradCAM(model, target_layer)
    
    for index in indices_to_plot:
        # بررسی معتبر بودن اندیس
        if index >= len(dataset):
            print(f"هشدار: اندیس {index} خارج از محدوده دیتاست است. (اندازه دیتاست: {len(dataset)})")
            continue
            
        # گرفتن تصویر و برچسب از دیتاست
        image_tensor, label_idx = dataset[index]
        input_tensor = image_tensor.unsqueeze(0)
        
        # تولید نقشه حرارتی
        heatmap = grad_cam(input_tensor)
        
        # مصورسازی
        original_img = tensor_to_img(input_tensor.clone())
        cam_image = show_cam_on_image(original_img, heatmap)
        
        # نمایش با Matplotlib
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        true_label_name = dataset.classes[label_idx]
        fig.suptitle(f"Image Index: {index} - True Label: '{true_label_name}'", fontsize=16)

        ax1.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image')
        ax1.axis('off')

        ax2.imshow(cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB))
        ax2.set_title('Grad-CAM')
        ax2.axis('off')
        
        plt.show()


# --- بخش ۳: مثال نحوه استفاده ---
if __name__ == '__main__':
    # =========================================================================
    # ۱. آماده‌سازی مدل و دیتاست (این بخش‌ها را با مدل و دیتاست خود جایگزین کنید)
    # =========================================================================

    # ایجاد یک دیتاست ساختگی برای نمونه
    data_dir = "my_custom_dataset"
    if not os.path.exists(data_dir):
        print("ایجاد دیتاست ساختگی...")
        os.makedirs(os.path.join(data_dir, "bulldog"))
        os.makedirs(os.path.join(data_dir, "tabby_cat"))
        # چند تصویر نمونه ایجاد می‌کنیم
        Image.new('RGB', (224, 224), color = 'red').save(os.path.join(data_dir, "bulldog", "dog_0.png"))
        Image.new('RGB', (224, 224), color = 'green').save(os.path.join(data_dir, "bulldog", "dog_1.png"))
        Image.new('RGB', (224, 224), color = 'blue').save(os.path.join(data_dir, "tabby_cat", "cat_0.png"))

    # تعریف تبدیلات و بارگذاری دیتاست
    # آدرس دیتاست خود را اینجا قرار دهید
    my_dataset_path = data_dir 
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    my_dataset = ImageFolder(root=my_dataset_path, transform=transform)
    
    # مدل خود را اینجا بارگذاری کنید
    my_model = resnet18(weights='IMAGENET1K_V1')

    # =========================================================================
    # ۲. اجرای تحلیل
    # =========================================================================

    # پیدا کردن خودکار آخرین لایه کانولوشنی
    last_conv_layer = find_last_conv_layer(my_model)

    # لیست شماره تصاویری که می‌خواهید تحلیل شوند را اینجا وارد کنید
    indices_to_check = [0, 2] # تصویر اول (سگ) و سوم (گربه) را تحلیل می‌کنیم

    # فراخوانی تابع اصلی برای شروع تحلیل
    analyze_and_plot_indices(
        model=my_model,
        target_layer=last_conv_layer,
        dataset=my_dataset,
        indices_to_plot=indices_to_check
    )