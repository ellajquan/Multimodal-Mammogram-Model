import os
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
from skimage import io
import json
from io import BytesIO

# Load configuration file
with open('config.json') as config_file:
    config = json.load(config_file)

# Configuration settings
file_path_excel = config.get("file_path_excel")
data_percentage = config.get("data_percentage", 1)  # Default to 100% if not set
output_dir = config.get("output_dir")
views = config.get("views", ["CC", "MLO", "US"])  # Default views if not set
output_excel_name = config.get("output_excel_name", "test_labels.xlsx")

# Load metadata
metadata = pd.read_excel(file_path_excel)

# Sample a subset of the data based on data_percentage
metadata = metadata.sample(frac=data_percentage, random_state=42).reset_index(drop=True)

# Create output directories
os.makedirs(output_dir, exist_ok=True)
out_dir = os.path.join(output_dir, 'test')
os.makedirs(out_dir, exist_ok=True)

excel_data = []

def convert_dcm_to_png(dcm_file_path, output_png_path):
    """Convert a DICOM file to PNG and save."""
    try:
        dcm = pydicom.dcmread(dcm_file_path)
        pixel_array = dcm.pixel_array
        plt.imsave(output_png_path, pixel_array, cmap='gray')
    except Exception as e:
        print(f"Error converting {dcm_file_path}: {e}")

# Process each row in the metadata and convert files as per configuration
for _, row in metadata.iterrows():
    patient_id = row['patient_id'].replace('D2-', '')
    subtype = row['subtype']
    
    # Process each view in config
    for view in views:
        file_path = row.get('file_path')  # Get the file path from the metadata
        
        if pd.isna(file_path):
            print(f"No file path found for {view} view in patient {patient_id}. Skipping.")
            continue

        # Define the output path for the PNG file
        output_png_path = os.path.join(out_dir, f'{patient_id}_{view}.png')
        convert_dcm_to_png(file_path, output_png_path)
        
        # Add entry for the new excel
        entry = {
            'patient_id': patient_id,
            f'{view}_file': output_png_path,  # Dynamically assigns view-specific file paths
            'subtype': subtype
        }
        excel_data.append(entry)

# Create DataFrame and pivot for required excel format
df = pd.DataFrame(excel_data)
df_pivot = df.pivot_table(index='patient_id', values=[f'{view}_file' for view in views], aggfunc='first').reset_index()
df_pivot['subtype'] = metadata.groupby('patient_id')['subtype'].first().values
df_pivot['target'] = df_pivot['subtype'].apply(lambda x: 1 if x == "Luminal A" else 0)

# Save the excel file based on config setting
output_excel_path = os.path.join(output_dir, output_excel_name)
df_pivot.to_excel(output_excel_path, index=False)

print(f"DICOM files converted and {output_excel_name} saved in {output_dir}.")
