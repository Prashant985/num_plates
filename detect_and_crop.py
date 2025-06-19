from ultralytics import YOLO
import cv2
import os
from pathlib import Path

# Initialize
model = YOLO('/Users/prashantshah/Desktop/num_plates/runs/detect/train/weights/best.pt')
input_dir = '/Users/prashantshah/Desktop/num_plates/images/train'  # Your image directory
output_dir = '/Users/prashantshah/Desktop/num_plates/cropped_plates'  # Where to save plates
conf_threshold = 0.5  # Minimum confidence

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Process all images
for img_path in Path(input_dir).glob('*.jpg'):
    # Run prediction
    results = model.predict(str(img_path), conf=conf_threshold)
    
    # Process results
    for i, result in enumerate(results):
        if len(result.boxes) == 0:
            continue
            
        # Save cropped plates
        for j, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            plate = result.orig_img[y1:y2, x1:x2]
            
            # Generate unique filename
            base_name = f"{img_path.stem}_plate_{j}.jpg"
            output_path = os.path.join(output_dir, base_name)
            
            # Save if plate is reasonably sized
            if plate.size > 1000:  # At least 30x30 pixels
                cv2.imwrite(output_path, plate)
                print(f"Saved {output_path}")

print(f"\nProcessing complete! Cropped plates saved to {output_dir}")
