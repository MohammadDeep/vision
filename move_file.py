import shutil
import os

def move_all_files(src_dir, dest_dir):
    # بررسی وجود دایرکتوری مبدا
    if os.path.exists(src_dir):
        # بررسی وجود دایرکتوری مقصد
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)  # اگر دایرکتوری مقصد وجود نداشت، آن را بساز

        # جابجایی تمام فایل‌ها
        for filename in os.listdir(src_dir):
            file_path = os.path.join(src_dir, filename)
            text = filename
            parts = text.split('-')
            file_name1 = parts[-1]
            # اگر فایل باشد (نه دایرکتوری)
            if os.path.isfile(file_path):
                shutil.move(file_path, os.path.join(dest_dir, file_name1))
                print(f"File {filename} moved to {dest_dir}/{file_name1}")
            else:
                print(f"{file_path} is a directory, skipping.")
    else:
        print(f"Source directory {src_dir} does not exist.")

# مثال استفاده:
src_directory = input( 'dase dir')
dest_directory = input('final dir')

move_all_files(src_directory, dest_directory)
