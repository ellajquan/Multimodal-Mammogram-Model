import os
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
from skimage import io
import json
from sklearn.model_selection import train_test_split

# Load configuration file
with open('config.json') as config_file:
    config = json.load(config_file)

# Configuration settings
file_path_excel = config.get("file_path_excel")
data_percentage = config.get("data_percentage", 1)  # Default to 100% if not set
train_test_split_percentage = config.get("train_test_split_percentage", 0.8)  # Default to 80% train, 20% test
output_dir_train = config.get("output_dir_train")
output_dir_test = config.get("output_dir_test")
views = config.get("views", ["CC", "MLO", "US"])  # Default views if not set
output_excel_name_train = config.get("output_excel_name_train", "train_labels.xlsx")
output_excel_name_test = config.get("output_excel_name_test", "test_labels.xlsx")

# Load metadata
metadata = pd.read_excel(file_path_excel)

# Sample a subset of the data based on data_percentage
metadata = metadata.sample(frac=data_percentage, random_state=42).reset_index(drop=True)

# Split metadata into train and test sets
train_metadata, test_metadata = train_test_split(metadata, train_size=train_test_split_percentage, random_state=42)

# Create output directories
os.makedirs(output_dir_train, exist_ok=True)
os.makedirs(output_dir_test, exist_ok=True)

train_dir = os.path.join(output_dir_train, 'train')
test_dir = os.path.join(output_dir_test, 'test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

def convert_dcm_to_png(dcm_file_path, output_png_path):
    """Convert a DICOM file to PNG and save."""
    try:
        dcm = pydicom.dcmread(dcm_file_path)
        pixel_array = dcm.pixel_array
        plt.imsave(output_png_path, pixel_array, cmap='gray')
    except Exception as e:
        print(f"Error converting {dcm_file_path}: {e}")

def process_metadata(metadata, output_dir):
    csv_data = []
    for _, row in metadata.iterrows():
        patient_id = row['patient_id'].replace('D2-', '').zfill(4)  # Add zero padding to make it four digits
        subtype = row['subtype']
        
        # Process each view in config
        for view in views:
            file_path = row.get('file_path')  # Get the file path from the metadata
            
            if pd.isna(file_path):
                print(f"No file path found for {view} view in patient {patient_id}. Skipping.")
                continue

            # Define the output path for the PNG file
            output_png_path = os.path.join(output_dir, f'{patient_id}_{view}.png')
            convert_dcm_to_png(file_path, output_png_path)
            
            # Add entry for the new DataFrame
            entry = {
                'patient_id': patient_id,
                f'{view}_file': output_png_path,  # Dynamically assigns view-specific file paths
                'subtype': subtype
            }
            csv_data.append(entry)
    
    return csv_data

# Process and save train metadata
train_csv_data = process_metadata(train_metadata, train_dir)
df_train = pd.DataFrame(train_csv_data)
df_train_pivot = df_train.pivot_table(index='patient_id', values=[f'{view}_file' for view in views], aggfunc='first').reset_index()
df_train_pivot['subtype'] = train_metadata.groupby('patient_id')['subtype'].first().values
df_train_pivot['target'] = df_train_pivot['subtype'].apply(lambda x: 1 if x == "Luminal A" else 0)
output_excel_path_train = os.path.join(output_dir_train, output_excel_name_train)
df_train_pivot.to_excel(output_excel_path_train, index=False)

# Process and save test metadata
test_csv_data = process_metadata(test_metadata, test_dir)
df_test = pd.DataFrame(test_csv_data)
df_test_pivot = df_test.pivot_table(index='patient_id', values=[f'{view}_file' for view in views], aggfunc='first').reset_index()
df_test_pivot['subtype'] = test_metadata.groupby('patient_id')['subtype'].first().values
df_test_pivot['target'] = df_test_pivot['subtype'].apply(lambda x: 1 if x == "Luminal A" else 0)
output_excel_path_test = os.path.join(output_dir_test, output_excel_name_test)
df_test_pivot.to_excel(output_excel_path_test, index=False)

print(f"DICOM files converted and saved in {output_dir_train} for training and {output_dir_test} for testing.")
