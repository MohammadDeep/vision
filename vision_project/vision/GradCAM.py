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

# کلاس GradCAM که منطق اصلی را پیاده‌سازی می‌کند
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # ثبت هوک‌ها
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_tensor, target_class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if target_class_idx is None:
            target_class_idx = output.argmax(dim=1).item()

        # گذر پس‌رو از کلاس هدف
        self.model.zero_grad()
        output[0, target_class_idx].backward()

        # محاسبه وزن‌ها و نقشه حرارتی
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()

        # وزن‌دهی به نقشه‌های فعال‌سازی
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = nn.functional.relu(heatmap)
        
        # نرمال‌سازی برای نمایش بهتر
        heatmap /= torch.max(heatmap)
        return heatmap.cpu().numpy()

# تابع کمکی برای نمایش نقشه حرارتی روی تصویر
def show_cam_on_image(img, heatmap):
    # تبدیل نقشه حرارتی به فرمت رنگی
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # ترکیب تصویر و نقشه حرارتی
    superimposed_img = heatmap * 0.4 + img * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    return superimposed_img

# تابع برای تبدیل تنسور به تصویر قابل نمایش
def tensor_to_img(tensor):
    # Un-normalize
    tensor = tensor.squeeze(0)
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    img = inv_normalize(tensor)
    img = img.permute(1, 2, 0)
    img = (img.numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # تبدیل به BGR برای OpenCV

# --- اجرای اصلی ---
if __name__ == '__main__':
    # 1. ایجاد دیتاست ساختگی (شما با دیتاست خود جایگزین کنید)
    data_dir = "gradcam_demo_dataset"
    if not os.path.exists(data_dir):
        print("ایجاد دیتاست ساختگی...")
        os.makedirs(os.path.join(data_dir, "dog"))
        os.makedirs(os.path.join(data_dir, "cat"))
        # می‌توانید تصاویر واقعی سگ و گربه را اینجا کپی کنید
        # در اینجا از تصاویر رنگی ساده استفاده می‌کنیم
        Image.new('RGB', (224, 224), color = 'red').save(os.path.join(data_dir, "dog", "dog1.png"))
        Image.new('RGB', (224, 224), color = 'blue').save(os.path.join(data_dir, "cat", "cat1.png"))

    # 2. بارگذاری دیتاست و انتخاب یک تصویر تصادفی
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageFolder(root=data_dir, transform=transform)
    random_idx = random.randint(0, len(dataset) - 1)
    image_tensor, label = dataset[random_idx]
    input_tensor = image_tensor.unsqueeze(0)

    # 3. بارگذاری مدل و انتخاب لایه هدف
    model = resnet18(weights='IMAGENET1K_V1')
    target_layer = model.layer4[-1].conv2

    # 4. ایجاد شیء GradCAM و تولید نقشه حرارتی
    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam(input_tensor)

    # 5. مصورسازی نهایی
    original_img = tensor_to_img(input_tensor.clone())
    cam_image = show_cam_on_image(original_img, heatmap)

    # نمایش با Matplotlib
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2.imshow(cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB))
    ax2.set_title('Grad-CAM')
    ax2.axis('off')
    
    # نمایش کلاس پیش‌بینی شده
    model.eval()
    pred_idx = model(input_tensor).argmax(dim=1).item()
    # این کد نیاز به فایل کلاس‌های ImageNet دارد، اما می‌توانیم فقط اندیس را نمایش دهیم
    fig.suptitle(f"Random Image (label: {dataset.classes[label]}), Predicted Index: {pred_idx}", fontsize=16)

    plt.show()