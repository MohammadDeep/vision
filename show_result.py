from Config import dir_history_model
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm
import os


#Create dir for save 
csv_files = [f.name for f in dir_history_model.glob("*.csv")]
dir_save = Path(dir_history_model,'create epoch')
if not os.path.exists(dir_save):
    os.makedirs(dir_save)

# for all csv file in baser dir
for file_name in tqdm(csv_files):
    
    dir_file = Path(dir_history_model, file_name)
    file = pd.read_csv(dir_file)
    if 'number/ len_data' in file.columns:
        list_data = [ i.split('/') for i in file['number/ len_data']]
        list_n = [int(j[0])/int(j[1]) for j in list_data]
        n_list = [0]
        for number in list_n:
            n_list.append(number)
            if number == 1 :
                break
        

        df_n_list = np.array(n_list[1:])-np.array(n_list[:-1])
        file.drop('number/ len_data', axis=1 , inplace=True)

        for i in range(len(file)):
            file.iloc[i] = file.iloc[i] * df_n_list[(i + 1 ) % len(df_n_list)]
        x = None
        if len(file) % len(df_n_list) != 0: 
            print(f'{len(file)} % {len(df_n_list)}  = {len(file) % len(df_n_list)}') 
            print(f'epoch not complet for file {file_name}')  
            x = input('type error to add in mane')
            number = int((len(file) // len(df_n_list)) * len(df_n_list))
            file = file[: number]
        else:
            x = None
        file_new = file.groupby(file.index // len(df_n_list)).sum()
        file_new["epoches"] = range(1, len(file_new) + 1)
        # جابجا کردن ستون‌ها: epoches اول، بقیه بعدش
        cols = ["epoches"] + [col for col in file_new.columns if col != "epoches"]
        file_new = file_new[cols]
        file_new.to_csv(Path(dir_save, f'{x}_{file_name}'), index=False)
            
        

