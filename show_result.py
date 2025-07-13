from Config import dir_history_model
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm
#print(dir_history_model)

csv_files = [f.name for f in dir_history_model.glob("*.csv")]

#print(csv_files)

for file_name in tqdm(csv_files):
    
    dir_file = Path(dir_history_model, file_name)
    file = pd.read_csv(dir_file)
    if 'number/ len_data' in file.columns:
        #print(file.index[0])
        list_data = [ i.split('/') for i in file['number/ len_data']]
        list_n = [int(j[0])/int(j[1]) for j in list_data]
        n_list = [0]
        for number in list_n:
            n_list.append(number)
            if number == 1 :
                break
        #print(n_list)

        df_n_list = np.array(n_list[1:])-np.array(n_list[:-1])
        file.drop('number/ len_data', axis=1 , inplace=True)

        for i in range(len(file)):
            file.iloc[i] = file.iloc[i] * df_n_list[(i + 1 ) % len(df_n_list)]
        file = file[: (file.index // len(df_n_list)) * len(df_n_list)]
        file_new = file.groupby(file.index // len(df_n_list)).sum()
        file_new["epoches"] = range(1, len(file_new) + 1)
        # جابجا کردن ستون‌ها: epoches اول، بقیه بعدش
        cols = ["epoches"] + [col for col in file_new.columns if col != "epoches"]
        file_new = file_new[cols]
        file_new.to_csv(Path(dir_history_model, file_name), index=False)
            
        
        #print(df_n_list) 
        

