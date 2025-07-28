from vision.Create_Dataset import Create_dataset
from vision.Config import dir_extended_cut_image_non_core_class_val,dir_extended_all_folber,dir_extended_all_folber_val, dic_dir_val,dic_dir,dir_dataset_folder,dir_dataset_folder_val, Create_train_data, Create_val_data
from pathlib import Path


if Create_train_data:
    id_class = 1
    dic_name_folber_clases = {
        
        f'class_id{id_class}':['core_class' ,'cut_image' , 'box_image'],
        'calss_other':['non_core_class', 'bake_image_Not']

        }

  
    Create_dataset (dic_name = dic_name_folber_clases ,
                         dic_dir = dic_dir,
                        dir_extended = dir_extended_all_folber,
                        move_or_copy_files = 'move'
                        )
    

if Create_val_data:
    id_class = 1
    dic_name_folber_clases_val = {
        
        f'class_id{id_class}':['core_class' ,'cut_image' , 'box_image'],
        'calss_other':['non_core_class', 'bake_image_Not']

        }

    

    Create_dataset (dic_name = dic_name_folber_clases_val ,
                        dic_dir = dic_dir_val,
                        dir_extended = dir_extended_all_folber_val,
                        move_or_copy_files = 'move'
                        )
    
    dic_name_folber_clases_val = {
        
        f'class_id{id_class}':['core_class' ],
        'calss_other':[ 'bake_image_Not']

        }

    

    Create_dataset (dic_name = dic_name_folber_clases_val ,
                        dic_dir = dic_dir_val,
                        dir_extended = dir_extended_cut_image_non_core_class_val,
                        move_or_copy_files = 'move'
                        )