from vision.Config import dir_extended_dataset,dir_extended_dataset_val ,dic_dir , dic_dir_val, dic_name_folber_clases , dic_name_folber_clases_val
from vision.train_val_functiones.im_show import move_or_copy_images_to_folder, tree_directory_images
from pathlib import Path
def Create_dataset (dic_name = dic_name_folber_clases ,
                    dic_dir = dic_dir,
                    dir_extended = dir_extended_dataset,
                    move_or_copy_files = 'move'
                    ):
    for class_name in dic_name:
        for folber_name in dic_name[class_name]:
            print(f'{move_or_copy_files} : {dic_dir[folber_name]} -> {Path(dir_extended, class_name)}')
            move_or_copy_images_to_folder(
                source_folder = dic_dir[folber_name],
                destination_folder = Path(dir_extended, class_name),
                pasvand = folber_name,
                move_or_copy= move_or_copy_files
            )

    tree_directory_images(str(dir_extended))

