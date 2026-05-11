import os
import random
from PIL import Image
import matplotlib.pyplot as plt

# Use LOCAL path instead of Kaggle path
train_path = "/Users/prashantshah/Desktop/num_plates/split_dataset/train"
images_per_class = 2  # Number of sample images to display per class

# Check if train directory exists
if not os.path.exists(train_path):
    raise FileNotFoundError(f"Train directory not found at: {train_path}")

# Get all classes (only directories)
classes = [d for d in os.listdir(train_path) 
          if os.path.isdir(os.path.join(train_path, d))]

if not classes:
    raise ValueError("No class directories found in the training set")

# Create figure with appropriate size
num_classes = len(classes)
fig, axes = plt.subplots(
    num_classes, 
    images_per_class, 
    figsize=(12, num_classes * 2),
    squeeze=False  # Ensures axes is always 2D array
)

# Plot sample images for each class
for i, class_name in enumerate(sorted(classes)):  # Sort classes alphabetically
    class_path = os.path.join(train_path, class_name)
    
    try:
        images = [f for f in os.listdir(class_path) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not images:
            print(f"Warning: No images found in class {class_name}")
            continue
            
        random.shuffle(images)
        
        for j in range(min(images_per_class, len(images))):  # Handle cases with <2 images
            try:
                img_path = os.path.join(class_path, images[j])
                img = Image.open(img_path)
                
                ax = axes[i, j]
                ax.imshow(img)
                ax.axis("off")
                
                if j == 0:  # Label only the first column
                    ax.set_title(f"{class_name} ({len(images)} images)", 
                               fontsize=10, 
                               loc='left', 
                               pad=2)
                    
            except Exception as e:
                print(f"Error loading {img_path}: {str(e)}")
                # Display blank if image fails to load
                axes[i, j].axis("off")
                axes[i, j].text(0.5, 0.5, "Image Error", 
                               ha='center', va='center')
                
    except Exception as e:
        print(f"Error processing class {class_name}: {str(e)}")
        for j in range(images_per_class):
            axes[i, j].axis("off")

plt.tight_layout(pad=1.0)
plt.suptitle(f"Training Set Samples (showing {images_per_class} random samples per class)", y=1.02)
plt.show()
