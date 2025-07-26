import os
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import os
import torch
from torchvision import models





from vision.Config import dir_history_model_google_dirve ,dir_extended_all_folber_val
from vision.train_val_functiones.train_val_functiones import load_model, test_step

def find_files_by_content(folder_path, file_extension, start_name):
    matched_files = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(file_extension) and file.startswith(start_name):
                file_path = os.path.join(root, file)
                matched_files.append(file_path)
               
    return matched_files

def test_model(
        dic_model,
        list_dataset_dir = [dir_extended_all_folber_val],
        model_loss_funciont = nn.BCEWithLogitsLoss(),
        model_save_dir = dir_history_model_google_dirve ,
        BATCH_SIZE = 2 ** 7,
        extension = '.pth',
        text_to_find = None
         ):
    

    model_name= dic_model['model_name']
    model_stucher = dic_model['model_stucher']
    input_shape =dic_model['input_shape']
    mean = dic_model['mean']
    std = dic_model['std']

    if text_to_find is None:
        text_to_find = model_name
    dires_model = find_files_by_content(model_save_dir, extension, text_to_find)
    
    dic_result ={}

    for dataset_dir  in list_dataset_dir:

        transform_val = transforms.Compose([
        transforms.Resize(input_shape),   # تغییر اندازه به ورودی مدل
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
        ])


        # ImageFolder دو کلاس را 0 و 1 نگاشت می‌کند
        dataset_val = datasets.ImageFolder(root=dataset_dir, transform=transform_val)
        dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)







        for dir_model in dires_model:
            print(f'test model save in dir -> {dir_model}')
            print('loadin model:')
            model_load = load_model(
                    model_stucher,
                    dir_model
                )
            
            
            loss, acc , F1, Tensor_T_F= test_step(model_load,
                            dataloader_val,
                            model_loss_funciont,
                            show_plot_and_F1 = True)
                        
            dic_result[f'{dir_model}'] = {
                'dir_dataset':dataset_dir,
                'loss' : loss,
                'accuracy': acc,
                'F1' : F1, 
                'TT': Tensor_T_F[0],
                'FT': Tensor_T_F[1],
                'TF': Tensor_T_F[2],
                'FF': Tensor_T_F[3],
                }
            


    return dic_result