import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

# Initialize models
model = YOLO('yolov8n.pt')
reader = easyocr.Reader(['ne', 'en'])  # Nepali + English

def process_plate_detection(image_path='Biketrain2_plate_0.jpg'):  # <-- Add parameter with default value
    """Process number plate detection on an image"""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return
    
    print(f"\nProcessing image: {image_path}")
    print(f"Original dimensions: {img.shape} (HxWxC)")
    
    # 1. Detect potential plates
    results = model(img, conf=0.5)  # Medium confidence
    
    for i, result in enumerate(results):
        boxes = result.boxes.xyxy.cpu().numpy()
        print(f"\nFound {len(boxes)} potential plate(s)")
        
        for j, box in enumerate(boxes, 1):
            x1, y1, x2, y2 = map(int, box)
            print(f"\nPlate {j} coordinates: ({x1}, {y1}) to ({x2}, {y2})")
            
            # 2. Crop plate region
            plate_img = img[y1:y2, x1:x2]
            cv2.imwrite(f'plate_{j}_crop.jpg', plate_img)
            
            # 3. Preprocess for OCR
            plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            _, plate_thresh = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cv2.imwrite(f'plate_{j}_processed.jpg', plate_thresh)
            
            # 4. OCR with Nepali allowlist
            ocr_result = reader.readtext(
                plate_thresh,
                allowlist='ककोखगचजझञडितनापप्रबबाभेममेयलुसीसुसेह०१२३४५६७८९',
                width_ths=3.0,
                text_threshold=0.4
            )
            
            if ocr_result:
                for (_, text, prob) in ocr_result:
                    print(f"Detected: '{text}' (confidence: {prob:.2f})")
            else:
                print("No text detected - check processed images")
                # In your plate_reader.py after text extraction
ocr_text = "बा१२च१२३४"  # Raw OCR result
validation = process_plate_detection(ocr_text)

if validation['valid']:
    print(f"Valid {validation['type']} plate: {validation['standardized']}")
else:
    print(f"Invalid plate: {validation['error']}")


# Example usage with your image
process_plate_detection('/Users/prashantshah/Desktop/num_plates/cropped_plates/Biketrain8_plate_0.jpg')  # Replace with your actual image path