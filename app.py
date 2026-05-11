from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'static/results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Define the character classifier model
class NepaliPlateCharacterClassifier(nn.Module):
    def __init__(self):
        super(NepaliPlateCharacterClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Adjust based on your input size
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 34)  # 52 classes for Nepali characters/digits

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load models
plate_detector = YOLO('/Users/prashantshah/Desktop/num_plates/runs/detect/train/weights/best.pt')

# Initialize character classifier
character_classifier = NepaliPlateCharacterClassifier()

# Load pretrained weights with error handling
try:
    character_classifier.load_state_dict(torch.load('trained_model.pth'))
except Exception as e:
    print(f"Error loading model weights: {e}")
    print("Trying partial weight loading...")
    pretrained_dict = torch.load('trained_model.pth')
    model_dict = character_classifier.state_dict()
    
    # 1. Filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    
    # 2. Overwrite entries in existing state dict
    model_dict.update(pretrained_dict)
    
    # 3. Load the new state dict
    character_classifier.load_state_dict(model_dict, strict=False)

character_classifier.eval()

# Nepali character classes
nepali_characters = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'ब', 'भ', 'प', 'त', 'थ', 'द', 'ध', 'न', 'म', 'य',
    'र', 'ल', 'व', 'श', 'स', 'ष', 'ह', 'अ', 'आ', 'इ',
    'ई', 'उ', 'ऊ', 'ए', 'ऐ', 'ओ', 'औ', 'क', 'ख', 'ग',
    'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड',
    'ढ', 'ण'
]

# Image transforms
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Must match original training size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # For 3-channel RGB
])


def enhance_image(img):
    """Improve image quality for better detection"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced_l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((enhanced_l, a, b)), cv2.COLOR_LAB2BGR)

def classify_character(char_img):
    """Classify individual character using the Nepali classifier"""
    try:
        char_pil = Image.fromarray(char_img)
        char_tensor = transform(char_pil).unsqueeze(0)
        
        with torch.no_grad():
            outputs = character_classifier(char_tensor)
            _, predicted = torch.max(outputs.data, 1)
        
        return nepali_characters[predicted.item()]
    except Exception as e:
        print(f"Character classification error: {e}")
        return None

def segment_and_recognize(plate_img):
    """Segment plate characters and classify them"""
    # Preprocessing
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    chars = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 15:  # Filter small noise
            char_img = gray[y:y+h, x:x+w]
            char_img = cv2.resize(char_img, (32, 32))
            _, char_img = cv2.threshold(char_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            char_class = classify_character(char_img)
            if char_class:
                chars.append((x, char_class))
    
    # Sort characters left-to-right
    chars.sort(key=lambda x: x[0])
    plate_text = ''.join([c[1] for c in chars])
    
    return plate_text if plate_text else None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load and enhance image
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'success': False, 'error': 'Invalid image file'}), 400
        
        enhanced_img = enhance_image(img)
        
        # Detect plates
        results = plate_detector(enhanced_img, conf=0.5)
        plates = []
        
        for result in results:
            for box in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                
                # Expand plate area
                h, w = img.shape[:2]
                x1 = max(0, x1 - int(0.1*(x2-x1)))
                y1 = max(0, y1 - int(0.1*(y2-y1)))
                x2 = min(w, x2 + int(0.1*(x2-x1)))
                y2 = min(h, y2 + int(0.1*(y2-y1)))
                
                plate_img = enhanced_img[y1:y2, x1:x2]
                
                # Recognize characters
                plate_text = segment_and_recognize(plate_img)
                
                # Draw visualization
                color = (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                
                if plate_text:
                    cv2.putText(img, plate_text, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                plates.append({
                    'coordinates': [x1, y1, x2, y2],
                    'text': plate_text,
                    'confidence': 0.95  # Placeholder, adjust as needed
                })
        
        # Save result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"result_{timestamp}.jpg"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        cv2.imwrite(result_path, img)
        
        return jsonify({
            'success': True,
            'plates': plates,
            'resultUrl': f'/static/results/{result_filename}'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/static/results/<filename>')
def serve_result(filename):
    return send_from_directory('static/results', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)