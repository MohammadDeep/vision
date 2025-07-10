
from Datasets.Config import Create_train_data,Create_val_data,dir_dataset_folder_val,image_dir, image_val_dir,ann_file, ann_file_val, list_calsses, dir_dataset_folder
from perproses_image.Cut_Image import CutImage
from Datasets.create_folber_dataset import getDataFoalberCoco


# Create forlder dataset 
#train
if Create_train_data:
    train_data =  getDataFoalberCoco(
            image_dir  ,
            ann_file,
            list_calsses,
            dir_dataset_folder)

    train_data.create_dataset_folber()
## val
if Create_val_data:
    val_data =  getDataFoalberCoco(
            image_val_dir  ,
            ann_file_val,
            list_calsses,
            dir_dataset_folder_val)

    val_data.create_dataset_folber()

