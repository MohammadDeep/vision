import os
import shutil
import tqdm

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
      os.makedirs(dest_dir)
      print(f"دایرکتوری مقصد ایجاد شد: '{dest_dir}'")
    except OSError as e:
      print(f"خطا در ایجاد دایرکتوری مقصد: {e}")
      return # در صورت بروز خطا، از تابع خارج شو
  list_name_source_dirs = find_files_and_create_substring(source_dirs , search_string= search_string)
  # مرحله ۲: پیمایش در لیست دایرکتوری‌های مبدأ
  for src_dir in tqdm(list_name_source_dirs):
    source_file_path = os.path.join(source_dirs , src_dir)
    try:
        shutil.move(source_file_path, dest_dir)
        #print(f"فایل '{filename}' از '{src_dir}' به '{dest_dir}' منتقل شد.")
    except Exception as e:
        print(f"خطا در انتقال فایل '{source_file_path}': {e}")

