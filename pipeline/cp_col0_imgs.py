import csv
import os
import shutil
import pandas as pd

# Input and output folder paths
input_folder = r"C:/Users/linwang/Box/Root_Penetration/GWAS_SCREEN_AUG23/Images/Raw_Images"
output_folder = r"C:/Work"

# Read CSV file
df = pd.read_csv('./master_data.csv')

# Iterate through rows
for index, row in df.iterrows():
    # Source file path
    path = row['path']
    source_path = os.path.join(input_folder,path[7:])
    print(source_path)
    

    # Destination file path
    destination_path = os.path.join(output_folder, path[7:])
    

    # Create destination folder if not exists
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)

    # Copy the file
    try:
        shutil.copy2(source_path, destination_path)
        print(f"File {destination_path} copied successfully.")
    except FileNotFoundError:
        print(f"File {destination_path} not found in source folder.")
    except PermissionError:
        print(f"Permission error: Unable to copy file {destination_path}.")