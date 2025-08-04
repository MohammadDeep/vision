import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from vision.train_val_functiones.train_val_functiones import load_model

# ----------------- 2. تعریف و ثبت هر دو نوع هوک -----------------

def analis_model(dic_model,
                 dir_pth_file,
                 labels,
                 input_tensor,
                 loss_nf = nn.CrossEntropyLoss()
                 ):
    
    model_class = dic_model['model_stucher']

    model =  load_model(model_class, dir_pth_file)
    # دیکشنری برای نگهداری خروجی‌ها (activations) و گرادیان‌ها
    activations = {}
    gradients = {}

    # تابع هوک برای ذخیره خروجی‌ها (Forward Hook)
    def save_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    # تابع هوک برای ذخیره گرادیان‌ها (Backward Hook)
    def save_gradient(name):
        def hook(module, grad_input, grad_output):
            gradients[name] = grad_output[0].detach()
        return hook

    # ثبت هر دو هوک برای لایه‌های مورد نظر
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            # ثبت هوک برای عبور رو به جلو
            layer.register_forward_hook(save_activation(name))
            # ثبت هوک برای عبور پس‌رو
            layer.register_backward_hook(save_gradient(name))

    # ----------------- 3. اجرای یک پاس رو به جلو و پس‌رو -----------------
    #input_tensor = torch.randn(64, 3, input_size, input_size)
    #labels = torch.randint(0, calsse_n, (64,))
    output = model(input_tensor)
    adjusted_labels = labels.float().unsqueeze(1)
    loss = loss_nf(output, adjusted_labels)
    loss.backward()

    # ----------------- 4. پلات کردن خروجی‌ها و گرادیان‌ها -----------------
    # تعداد لایه‌هایی که هوک دارند
    num_layers = len(activations)
    fig, axes = plt.subplots(num_layers, 2, figsize=(12, num_layers * 4))
    fig.suptitle('Activations (Forward Pass) vs. Gradients (Backward Pass)', fontsize=16)

    # تبدیل دیکشنری به لیست برای حفظ ترتیب
    items = list(activations.items())

    for i in range(num_layers):
        layer_name = items[i][0]

        # پلات کردن خروجی‌ها (Activations)
        ax_act = axes[i, 0]
        act_data = activations[layer_name].cpu().numpy().flatten()
        ax_act.hist(act_data, bins=100, color='skyblue', label='Activations')
        ax_act.set_title(f'Activations: {layer_name}')
        ax_act.set_xlabel('Activation values')
        ax_act.set_ylabel('Frequency')
        ax_act.axvline(act_data.mean(), color='blue', linestyle='dashed', linewidth=2)


        # پلات کردن گرادیان‌ها
        ax_grad = axes[i, 1]
        grad_data = gradients[layer_name].cpu().numpy().flatten()
        ax_grad.hist(grad_data, bins=100, color='salmon', label='Gradients')
        ax_grad.set_title(f'Gradients: {layer_name}')
        ax_grad.set_xlabel('Gradient values')
        ax_grad.axvline(grad_data.mean(), color='red', linestyle='dashed', linewidth=2)


    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()