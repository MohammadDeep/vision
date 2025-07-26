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
import os
import pandas as pd




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
        text_to_find = None,
        file_path = 'test_model/test_all_model.csv'
         ):
    

    model_name= dic_model['model_name']
    model_stucher = dic_model['model_stucher']
    input_shape =dic_model['input_shape']
    mean = dic_model['mean']
    std = dic_model['std']

    columns= ['model_name',
                'dir_dataset',
                'loss',
                'accuracy',
                'F1' , 
                'TT',
                'FT',
                'TF',
                'FF',
                'sum_all_data']

    if text_to_find is None:
        text_to_find = model_name
    dires_model = find_files_by_content(model_save_dir, extension, text_to_find)
    

    file_path = Path(dir_history_model_google_dirve , file_path)
    print ('-' * 50)
    if os.path.exists(file_path):
        df_result = pd.read_csv(file_path)
        print("read file.")
        
        if df_result.columns.tolist() != columns:
            print("⚠️ Columns note True ")
            df_result = pd.DataFrame(columns= columns)
    else:
        print("⚠️ Create new dataframe ")
        df_result = pd.DataFrame(columns= columns)

    

    for dataset_dir  in list_dataset_dir:
        dataset_dir = Path(dataset_dir)
        transform_val = transforms.Compose([
        transforms.Resize(input_shape),   # تغییر اندازه به ورودی مدل
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
        ])


        # ImageFolder دو کلاس را 0 و 1 نگاشت می‌کند
        dataset_val = datasets.ImageFolder(root=dataset_dir, transform=transform_val)
        dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)







        for dir_model in dires_model:
            dir_model = Path(dir_model)
            condition = (df_result['model_name'] == dir_model.stem) & (df_result['dir_dataset'] == dataset_dir.stem)
            
            if df_result[condition].empty:
                print('-' * 50)
                print(f'test model save in dir -> {dir_model}')
                print('loadin model:')
                try:
                    model_load = load_model(
                            model_stucher,
                            dir_model
                        )
                
                
                    loss, acc , F1, Tensor_T_F= test_step(model_load,
                                    dataloader_val,
                                    model_loss_funciont,
                                    show_plot_and_F1 = True)
                    Tensor_T_F = Tensor_T_F.cpu()
                    sum_all_data_len = Tensor_T_F.sum()
                    df_result.loc[len(df_result)] = [
                        dir_model.stem,
                        dataset_dir.stem,
                        loss,
                        acc,
                        F1, 
                        Tensor_T_F[0][0].item()/sum_all_data_len.item(),
                        Tensor_T_F[0][1].item()/sum_all_data_len.item(),
                        Tensor_T_F[1][1].item()/sum_all_data_len.item(),
                        Tensor_T_F[1][0].item()/sum_all_data_len.item(),
                        sum_all_data_len.item()
                    ]
                except Exception as e:
                    print(f"⚠️ Error for model {dir_model.stem}: {e}")
            
    df_result = df_result.sort_values(by='model_name').reset_index(drop=True)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_result.to_csv(file_path, index=False)
    print(f'saveing file in dirctory :{file_path}')

    return df_result


