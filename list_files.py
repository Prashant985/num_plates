# Start a fresh Python session and run:
import os
p = '/Users/prashantshah/Desktop/num_plates/nepaliplatecharacterclassifier/character_ocr'

for folder in os.listdir(p):
    print("Folder:", folder)  # Ensure this is indented with 4 spaces
    folder_path = os.path.join(p, folder)
    
    if os.path.isdir(folder_path):
        count = 0
        for file in os.listdir(folder_path):
            print("  File:", file)
            count += 1
            if count == 2:
                break
