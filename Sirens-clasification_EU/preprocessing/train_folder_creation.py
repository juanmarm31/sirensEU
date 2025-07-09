# pip install setuptools --upgrade
# pip install wheel


import os
import shutil

# Define the mapping from contractions to country names
country_mapping = {
    'us': 'usa',
    'sp': 'spain',
    'ja': 'japan',
    'ch': 'china',
    'in': 'india',
    'it': 'italy',
    'ge': 'germany',
    'fr': 'france',
    'ca': 'canada'
}

# Input and output directories
input_dir = "/home/avdata/xtract/rodrjl/DATABASE/train-DW_Sirens_RealRecordings/klassifizierung-sirens_nosirens/sirens"
output_dir = "/home/avdata/xtract/rodrjl/DATABASE/train-DW_Sirens_RealRecordings/DW_Sirens_RealRecordings"

# Create output subdirectories if they do not exist
for folder in country_mapping.values():
    os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

# Process each file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.wav'):
        # Split the filename to get the country code
        parts = filename.split('_')
        if len(parts) >= 3:
            country_code = parts[1]
            # Check if the country code is in the mapping
            if country_code in country_mapping:
                # Get the corresponding country folder
                country_folder = country_mapping[country_code]
                # Construct the full file paths
                src_path = os.path.join(input_dir, filename)
                dest_path = os.path.join(output_dir, country_folder, filename)
                # Copy the file to the destination folder
                shutil.copy2(src_path, dest_path)
                print(f'Copied {filename} to {country_folder}')

print('File classification and copying completed.')
