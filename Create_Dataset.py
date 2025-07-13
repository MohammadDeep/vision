from Config import dir_extended_dataset,dir_extended_dataset_val ,dic_dir , dic_dir_val, dic_name_folber_clases , dic_name_folber_clases_val
from train_val_functiones.im_show import move_or_copy_images_to_folder, tree_directory_images
from pathlib import Path
for class_name in dic_name_folber_clases_val:
    for folber_name in dic_name_folber_clases_val[class_name]:
        print(f'move/copy : {dic_dir_val[folber_name]} -> {Path(dir_extended_dataset_val, class_name)}')
        move_or_copy_images_to_folder(
            source_folder = dic_dir_val[folber_name],
            destination_folder = Path(dir_extended_dataset_val, class_name),
            pasvand = folber_name
        )

tree_directory_images(str(dir_extended_dataset_val))

