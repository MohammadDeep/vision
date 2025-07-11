
from Datasets.Config import Create_train_data,Create_val_data,dir_dataset_folder_val,image_dir, image_val_dir,ann_file, ann_file_val, list_calsses, dir_dataset_folder
from perproses_image.Cut_Image import CutImage
from Datasets.create_folber_dataset import getDataFoalberCoco
from pathlib import Path

# Create forlder dataset 
#train
if Create_train_data:
        train_data =  getDataFoalberCoco(
                image_dir  ,
                ann_file,
                list_calsses,
                dir_dataset_folder)

        train_data.create_dataset_folber()

        #Producing new iamges

        imm_train =  CutImage(image_dir = image_dir,
                bake_dir = Path(dir_dataset_folder , '-1'),
                ann_file = ann_file,
                category_id = list_calsses[0],
                destination_dir = Path(dir_dataset_folder , 'CutImage'),
                cache_size= 10000000,
                memory_threshold = 95.0
                )
        
        # Cut box of images
        imm_train.create_cut_box_images( min_area=500, random_n = 1,random_size_box_cut=True)


        # Create bake grund
        imm_train.create_cut_box_bake()

        # Create persion and bakgrund random
        imm_train.create_cut_images( 10000 )
        
## val
if Create_val_data:
        val_data =  getDataFoalberCoco(
                image_val_dir  ,
                ann_file_val,
                list_calsses,
                dir_dataset_folder_val)

        val_data.create_dataset_folber()

        '''
        imm_val =  CutImage(image_dir = image_val_dir,
                bake_dir = Path(dir_dataset_folder_val, '-1'),
                ann_file = ann_file_val,
                category_id = list_calsses[0],
                destination_dir = Path(dir_dataset_folder_val, 'CutImage'),
                cache_size= 10000000,
                memory_threshold = 95.0
                )
                
        # Cut box of images
        imm_val.create_cut_box_images( min_area=500, random_n = 1,random_size_box_cut=True)

        # Create bake grund
        imm_val.create_cut_box_bake()

        # Create persion and bakgrund random
        imm_val.create_cut_images( 10000 )
        '''
