import cv2
import easyocr
from ultralytics import YOLO
import os

from test_detection import process_plate_detection

# Initialize
reader = easyocr.Reader(['en','ne'])  # English text recognition
plate_detector = YOLO('/Users/prashantshah/Desktop/num_plates/runs/detect/train/weights/best.pt')  # Your trained plate detector

def extract_plate_text(image_path):
    # 1. Detect Plates
    detections = plate_detector(image_path)
    
    # 2. Process Each Detection
    for i, result in enumerate(detections):
        img = result.orig_img
        for j, box in enumerate(result.boxes):
            # Get plate coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Crop plate
            plate_img = img[y1:y2, x1:x2]
            
            # 3. Preprocess for better OCR
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
            # 4. Extract Text
            ocr_results = reader.readtext(thresh)
            plate_text = " ".join([res[1] for res in ocr_results])
            
            print(f"Plate {j+1} Text: {plate_text}")
            ocr_text = "बा१२च१२३४"  # Raw OCR result
        validation = process_plate_detection(ocr_text)

        if validation['valid']:
            print(f"Valid {validation['type']} plate: {validation['standardized']}")
        else:
            print(f"Invalid plate: {validation['error']}")
            
            # 5. Visualize (optional)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, plate_text, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        
        # Save visualized image
        output_path = f"detected_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, img)
        print(f"Saved results to {output_path}")

# Usage
extract_plate_text("/Users/prashantshah/Desktop/num_plates/cropped_plates/Biketrain12_plate_0.jpg")