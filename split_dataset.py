import os
import shutil
from sklearn.model_selection import train_test_split

# Path to the dataset
data_path = "/Users/prashantshah/Desktop/num_plates/nepaliplatecharacterclassifier/character_ocr"

# Output paths for train, val, and test splits
output_base = "/Users/prashantshah/Desktop/num_plates/split_dataset"
train_path = os.path.join(output_base, "train")
val_path = os.path.join(output_base, "val")
test_path = os.path.join(output_base, "test")

# Create directories for train, val, and test sets
for path in [train_path, val_path, test_path]:
    os.makedirs(path, exist_ok=True)

# Iterate over each folder (class)
for folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder)
    if not os.path.isdir(folder_path):
        continue  # Skip non-directory files
    
    # List all files in the current class folder
    files = os.listdir(folder_path)
    
    # Split files into train, val, and test sets
    train_files, temp_files = train_test_split(files, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=1/3, random_state=42)
    
    # Ensure the target directories for this class exist
    for sub_path in [train_path, val_path, test_path]:
        os.makedirs(os.path.join(sub_path, folder), exist_ok=True)
    
    # Move files to their respective directories
    for file_name, dest_folder in zip(
        [train_files, val_files, test_files], 
        [train_path, val_path, test_path]
    ):
        for file in file_name:
            src = os.path.join(folder_path, file)
            dest = os.path.join(dest_folder, folder, file)
            shutil.copy(src, dest)  # Copy file to destination

print("Dataset split completed!")
