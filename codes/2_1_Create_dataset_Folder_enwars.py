import os
import shutil
from vision.train_val_functiones.im_show import tree_directory_images

def find_files_and_create_substring(directory, search_string):
  """
  این تابع در یک دایرکتوری مشخص، فایل‌هایی را که در نامشان یک رشته خاص وجود دارد، 
  پیدا کرده و از نام آن‌ها یک بخش را استخراج می‌کند.

  Args:
    directory (str): مسیر دایرکتوری مورد نظر.
    search_string (str): رشته‌ای که باید در نام فایل‌ها جستجو شود.

  Returns:
    list: لیستی از رشته‌های استخراج شده از نام فایل‌ها.
  """
  found_substrings = []
  for filename in os.listdir(directory):
    if search_string in filename:
      
      found_substrings.append(filename)
    
  return found_substrings




def move_files_from_directories(source_dirs,search_string, dest_dir):
  """
  این تابع فایل‌ها را از لیستی از دایرکتوری‌های مبدأ به یک دایرکتوری مقصد منتقل می‌کند.

  Args:
    source_dirs (list): لیستی از رشته‌ها که هر کدام مسیر یک دایرکتوری مبدأ هستند.
    dest_dir (str): رشته‌ای که مسیر دایرکتوری مقصد را مشخص می‌کند.
  """
  # مرحله ۱: بررسی و ایجاد دایرکتوری مقصد در صورت عدم وجود
  if not os.path.exists(dest_dir):
    try:
      print('try to create dirctory .....')
      os.makedirs(dest_dir)
      print(f"دایرکتوری مقصد ایجاد شد: '{dest_dir}'")
    except OSError as e:
      print(f"خطا در ایجاد دایرکتوری مقصد: {e}")
      return # در صورت بروز خطا، از تابع خارج شو
  list_name_source_dirs = find_files_and_create_substring(source_dirs , search_string= search_string)
  # مرحله ۲: پیمایش در لیست دایرکتوری‌های مبدأ
  for src_dir in list_name_source_dirs:
    source_file_path = os.path.join(source_dirs , src_dir)
    try:
        shutil.move(source_file_path, dest_dir)
        #print(f"فایل '{filename}' از '{src_dir}' به '{dest_dir}' منتقل شد.")
    except Exception as e:
        print(f"خطا در انتقال فایل '{source_file_path}': {e}")
val_train = input('enter val data or train data(V/T):')
dest_dir_dataset = input('enter dir dataset for enwars :')
tree_directory_images(dest_dir_dataset)
from vision.Config import list_calsses
list_class_folder = ['calss_other', f'class_id{list_calsses[0]}']
print('-' * 50)
if val_train == 'V':
    from vision.Config import dir_dataset_folder_val,dir_non_core_class_val,dir_core_class_val,dir_box_image_val,dir_bake_image_Not_val,dir_cut_image_val
    for dir in list_class_folder:
        dest_dir = os.path.join( dest_dir_dataset, dir)
        tree_directory_images(dir_dataset_folder_val)
        move_files_from_directories(dest_dir = dir_box_image_val,search_string='cut_area', source_dirs=dest_dir)
        move_files_from_directories(dest_dir = dir_bake_image_Not_val,search_string='bake_idx_', source_dirs=dest_dir)
        move_files_from_directories(dest_dir = dir_cut_image_val,search_string='cut_indx_', source_dirs=dest_dir)
        move_files_from_directories(dest_dir = dir_non_core_class_val,search_string='non_core_class', source_dirs=dest_dir)
        move_files_from_directories(dest_dir = dir_core_class_val,search_string='core_class', source_dirs=dest_dir)
        print('-' * 50)
        tree_directory_images(dir_dataset_folder_val)
elif val_train == 'T':
    from vision.Config import dir_dataset_folder,dir_non_core_class,dir_core_class,dir_box_image,dir_bake_image_Not,dir_cut_image
    for dir in list_class_folder:
        dest_dir = os.path.join( dest_dir_dataset, dir)
        tree_directory_images(dir_dataset_folder)
        move_files_from_directories(dest_dir = dir_box_image,search_string='cut_area', source_dirs=dest_dir)
        move_files_from_directories(dest_dir = dir_bake_image_Not,search_string='bake_idx_', source_dirs=dest_dir)
        move_files_from_directories(dest_dir = dir_cut_image,search_string='cut_indx_', source_dirs=dest_dir)
        move_files_from_directories(dest_dir = dir_non_core_class,search_string='non_core_class', source_dirs=dest_dir)
        move_files_from_directories(dest_dir = dir_core_class,search_string='core_class', source_dirs=dest_dir)
        print('-' * 50)
        tree_directory_images(dir_dataset_folder)



