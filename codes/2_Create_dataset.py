from vision.Create_Dataset import Create_dataset
from vision.Config import dic_dataset
from pathlib import Path


dic_data = dic_dataset['all_folber']

Create_dataset (dic_name = dic_data['dic_name'] ,
                        dic_dir = dic_data['dic_dir'],
                    dir_extended = dic_data['dir'],
                    move_or_copy_files = 'move'
                    )
    
