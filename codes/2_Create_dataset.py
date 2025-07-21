from vision.Create_Dataset import Create_dataset
from vision.Config import dic_dir_val,dic_dir,dir_dataset_folder,dir_dataset_folder_val, Create_train_data, Create_val_data
from pathlib import Path


if Create_train_data:
    id_class = 1
    dic_name_folber_clases = {
        
        f'class_id{id_class}':['core_class' ,'cut_image' , 'box_image'],
        'calss_other':['non_core_class', 'bake_image_Not']

        }

    dir_extended_dataset = Path(dir_dataset_folder , 'extended_dataset_all_folderes')

    Create_dataset (dic_name = dic_name_folber_clases ,
                         dic_dir = dic_dir,
                        dir_extended = dir_extended_dataset,
                        move_or_copy_files = 'move'
                        )
    

if Create_val_data:
    id_class = 1
    dic_name_folber_clases_val = {
        
        f'class_id{id_class}':['core_class' ,'cut_image' , 'box_image'],
        'calss_other':['non_core_class', 'bake_image_Not']

        }

    dir_extended_dataset = Path(dir_dataset_folder_val , 'extended_dataset_all_folderes')

    Create_dataset (dic_name = dic_name_folber_clases_val ,
                        dic_dir = dic_dir_val,
                        dir_extended = dir_extended_dataset,
                        move_or_copy_files = 'move'
                        )
    
    dic_name_folber_clases_val = {
        
        f'class_id{id_class}':['core_class' ],
        'calss_other':[ 'bake_image_Not']

        }

    dir_extended_dataset = Path(dir_dataset_folder_val , 'extended_dataset_bakeImageNot_and_coreClass_folderes')

    Create_dataset (dic_name = dic_name_folber_clases_val ,
                        dic_dir = dic_dir_val,
                        dir_extended = dir_extended_dataset,
                        move_or_copy_files = 'move'
                        )