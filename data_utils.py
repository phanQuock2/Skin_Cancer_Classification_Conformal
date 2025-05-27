import os
import pandas as pd
import shutil

def create_directory_structure(base_dir='base_dir'):
    train_dir = os.path.join(base_dir, 'train_dir')
    val_dir = os.path.join(base_dir, 'val_dir')

    class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
    for folder in [train_dir, val_dir]:
        os.makedirs(folder, exist_ok=True)
        for class_name in class_names:
            os.makedirs(os.path.join(folder, class_name), exist_ok=True)

    return train_dir, val_dir

def identify_duplicates(df_data):
    # Lọc các ảnh không bị trùng lesion_id
    df = df_data.groupby('lesion_id').count()
    df = df[df['image_id'] == 1].reset_index()
    unique_list = df['lesion_id'].tolist()

    df_data['duplicates'] = df_data['lesion_id'].apply(lambda x: 'no' if x in unique_list else 'yes')

    return df_data

