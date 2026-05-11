import os
import matplotlib.pyplot as plt

# Corrected path (using local directory instead of Kaggle)
split_base_path = "/Users/prashantshah/Desktop/num_plates/split_dataset"
subfolders = ["train", "val", "test"]

# Dictionary to store counts and ratios
split_counts = {split: {} for split in subfolders}
class_stats = {}

# Count files in each split
for split in subfolders:
    split_path = os.path.join(split_base_path, split)
    if not os.path.exists(split_path):
        print(f"Warning: {split_path} does not exist!")
        continue
        
    for class_folder in os.listdir(split_path):
        class_path = os.path.join(split_path, class_folder)
        if os.path.isdir(class_path):
            try:
                file_count = len([f for f in os.listdir(class_path) 
                               if os.path.isfile(os.path.join(class_path, f))])
                split_counts[split][class_folder] = file_count
                
                # Initialize class stats if not exists
                if class_folder not in class_stats:
                    class_stats[class_folder] = {
                        'train': 0,
                        'val': 0,
                        'test': 0,
                        'total': 0
                    }
                class_stats[class_folder][split] = file_count
                class_stats[class_folder]['total'] += file_count
            except Exception as e:
                print(f"Error counting files in {class_path}: {e}")

# Print detailed class statistics
print("\nDetailed Class Distribution:")
for class_name, stats in class_stats.items():
    print(f"\nClass: {class_name}")
    print(f"  Total: {stats['total']}")
    for split in subfolders:
        count = stats[split]
        ratio = count / stats['total'] if stats['total'] > 0 else 0
        print(f"  {split.capitalize()}: {count} ({ratio:.1%})")

# Calculate overall ratios
print("\nOverall Dataset Split:")
total_files = sum(stats['total'] for stats in class_stats.values())
overall_counts = {split: sum(split_counts[split].values()) for split in subfolders}

for split in subfolders:
    count = overall_counts[split]
    ratio = count / total_files if total_files > 0 else 0
    print(f"{split.capitalize()}: {count} ({ratio:.1%})")

# Visualization
if class_stats:
    # Plot class distribution
    plt.figure(figsize=(15, 5))
    
    # Class counts
    plt.subplot(1, 2, 1)
    classes = sorted(class_stats.keys())
    counts = [class_stats[c]['total'] for c in classes]
    plt.bar(classes, counts)
    plt.title('Class Distribution')
    plt.xlabel('Character Class')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=90)
    
    # Split ratios
    plt.subplot(1, 2, 2)
    split_labels = ['Train', 'Val', 'Test']  # Changed to match your folder names
    split_counts = [overall_counts['train'], overall_counts['val'], overall_counts['test']]
    plt.pie(split_counts, labels=split_labels, autopct='%1.1f%%')
    plt.title('Dataset Split Ratio')
    
    plt.tight_layout()
    plt.savefig(os.path.join(split_base_path, 'dataset_stats.png'))
    print("\nSaved visualization to dataset_stats.png")
else:
    print("\nNo data found to visualize.")
