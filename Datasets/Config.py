from pathlib import Path
# dirctory dataset 
base_dir = Path('/home/mohammad/code_PMC/dataset/coco')
image_dir = Path(base_dir, 'train2017')  
ann_file = Path(base_dir, 'annotations_trainval2017/annotations/instances_train2017.json')  
image_val_dir = Path(base_dir, 'val2017 (1)/val2017/')
ann_file_val = Path(base_dir,'annotations_trainval2017/annotations/instances_val2017.json')

# list of data classes for 
list_calsses = [1]

# dirctory for save dataset foder for classes
dir_dataset_folder = Path(base_dir, 'data_set_floders')
dir_dataset_folder_val = Path(base_dir, 'data_set_floders_val')


# urn code for Create data
Create_train_data = False
Create_val_data = True