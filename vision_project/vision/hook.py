import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ... (بقیه کد شما بدون تغییر باقی می‌ماند)

def analis_model_with_weight_grads(dic_model, dir_pth_file, labels, input_tensor, model, loss_nf=nn.CrossEntropyLoss(), load=True):
    # ... (بخش بارگذاری مدل و تعریف هوک forward مانند قبل)
    
    activations = {}
    def save_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            layer.register_forward_hook(save_activation(name))

    # --- اجرای پاس رو به جلو و پس‌رو ---
    output = model(input_tensor)
    adjusted_labels = labels.float().unsqueeze(1)
    loss = loss_nf(output, adjusted_labels)
    
    # ابتدا گرادیان‌ها را صفر می‌کنیم
    model.zero_grad()
    # سپس پاس پس‌رو را اجرا می‌کنیم
    loss.backward()

    # --- حالا به گرادیان وزن‌ها دسترسی داریم ---
    weight_gradients = {}
    for name, layer in model.named_modules():
        # بررسی می‌کنیم که لایه وزن داشته باشد و گرادیان آن محاسبه شده باشد
        if isinstance(layer, (nn.Conv2d, nn.Linear)) and layer.weight.grad is not None:
            weight_gradients[name] = layer.weight.grad.detach()

    # --- پلات کردن ---
    num_layers = len(activations)
    fig, axes = plt.subplots(num_layers, 2, figsize=(12, num_layers * 4))
    fig.suptitle('Activations vs. Weight Gradients', fontsize=16)

    items = list(activations.items())

    for i in range(num_layers):
        layer_name = items[i][0]
        
        # پلات کردن خروجی‌ها (Activations)
        ax_act = axes[i, 0]
        act_data = activations[layer_name].cpu().numpy().flatten()
        ax_act.hist(act_data, bins=100, color='skyblue')
        ax_act.set_title(f'Activations: {layer_name}')
        ax_act.set_xlabel('Activation values')
        
        # پلات کردن گرادیان‌های وزن
        if layer_name in weight_gradients:
            ax_grad = axes[i, 1]
            grad_data = weight_gradients[layer_name].cpu().numpy().flatten()
            ax_grad.hist(grad_data, bins=100, color='lightgreen')
            ax_grad.set_title(f'Weight Gradients: {layer_name}')
            ax_grad.set_xlabel('Gradient values')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()