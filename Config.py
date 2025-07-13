from pathlib import Path
# dirctory dataset 
base_dir = Path('./__dataset__coco')
image_dir = Path(base_dir, 'train2017')  
ann_file = Path(base_dir, 'annotations/instances_train2017.json')  
image_val_dir = Path(base_dir, 'val2017/')
ann_file_val = Path(base_dir,'annotations/instances_val2017.json')

# list of data classes for 
list_calsses = [1]

# dirctory for save dataset foder for classes
dir_dataset_folder = Path(base_dir, 'data_set_floders')
dir_dataset_folder_val = Path(base_dir, 'data_set_floders_val')
# folber names
''' 
save image in:
dir_dataset_folber|---> folber_name_bake_image_Not
                  |--->folber_name_box_image
                  |---> flober_name_cut_image

'''
folber_name_bake_image_Not = 'bake_image_Not'
folber_name_box_image = 'box_image'
folber_name_cut_image = 'cut_image'

dir_bake_iamge_Not_val = Path(dir_dataset_folder_val , folber_name_bake_image_Not)
dir_box_image_val = Path(dir_dataset_folder_val , folber_name_box_image)
dir_cut_image_val = Path(dir_dataset_folder_val , folber_name_cut_image)
dir_class_one_val = Path(dir_dataset_folder_val, str(list_calsses[0]))
dir_class_nimus_one_val = Path(dir_dataset_folder_val, '-1')

dir_bake_iamge_Not = Path(dir_dataset_folder , folber_name_bake_image_Not)
dir_box_image = Path(dir_dataset_folder , folber_name_box_image)
dir_cut_image = Path(dir_dataset_folder , folber_name_cut_image)
dir_class_one = Path(dir_dataset_folder, str(list_calsses[0]))
dir_class_nimus_one = Path(dir_dataset_folder, '-1')



# urn code for Create data
Create_train_data = False
Create_val_data = True