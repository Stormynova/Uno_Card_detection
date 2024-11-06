from flask import Flask, render_template, Response, request
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import io

app = Flask(__name__)

# Define the model path and load model
model_path = Path("UNO_Card_Detection_YOLOv8") / "train2" / "weights" / "best.pt"
model = YOLO(str(model_path))

# Dictionary mapping internal class names to human-readable labels
name_mapping = {
    # Blue cards
    "lblue-0": "blue 0",
    "lblue-1": "blue 1",
    "lblue-2": "blue 2", 
    "lblue-20": "blue Draw two",
    "lblue-3": "blue 3",
    "lblue-4": "blue 4",
    "lblue-5": "blue 5",
    "lblue-6": "blue 6",
    "lblue-7": "blue 7",
    "lblue-8": "blue 8",
    "lblue-9": "blue 9",
    "lbluerevers-20": "blue reverse",
    "lblueskip-20": "blue skip",
    # Special cards
    "lcolor-40": "Wild",
    "lcolor-400": "Wild draw 4",
    # Green cards
    "lgreen-0": "green 0",
    "lgreen-1": "green 1", 
    "lgreen-2": "green 2",
    "lgreen-20": "green Draw two",
    "lgreen-3": "green 3",
    "lgreen-4": "green 4",
    "lgreen-5": "green 5",
    "lgreen-6": "green 6",
    "lgreen-7": "green 7",
    "lgreen-8": "green 8",
    "lgreen-9": "green 9",
    "lgreenrevers-20": "green reverse",
    "lgreenskip-20": "green skip",
    # Red cards
    "lred-0": "red 0",
    "lred-1": "red 1",
    "lred-2": "red 2",
    "lred-20": "red Draw two", 
    "lred-3": "red 3",
    "lred-4": "red 4",
    "lred-5": "red 5",
    "lred-6": "red 6",
    "lred-7": "red 7",
    "lred-8": "red 8",
    "lred-9": "red 9",
    "lredrevers-20": "red reverse",
    "lredskip-20": "red skip",
    # Yellow cards
    "lyellow-0": "yellow 0",
    "lyellow-1": "yellow 1",
    "lyellow-2": "yellow 2",
    "lyellow-20": "yellow Draw two",
    "lyellow-3": "yellow 3",
    "lyellow-4": "yellow 4",
    "lyellow-5": "yellow 5",
    "lyellow-6": "yellow 6",
    "lyellow-7": "yellow 7",
    "lyellow-8": "yellow 8",
    "lyellow-9": "yellow 9",
    "lyellowrevers-20": "yellow reverse",
    "lyellowskip-20": "yellow skip",
}
        

def process_image(image):
    """Process uploaded image and return annotated image with detections"""
    # Convert PIL Image to OpenCV format
    img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Run detection
    results = model(img_array)
    
    # Process and annotate detections
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box
            original_label = model.names[int(class_id)]
            mapped_label = name_mapping.get(original_label, original_label)
            label = f"{mapped_label}: {conf:.2f}"
            cv2.rectangle(img_array, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img_array, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Convert back to RGB for PIL
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_array)

def generate_frames():
    """Generate frames from webcam with UNO card detection"""
    cap = cv2.VideoCapture(0)  # 0 means the default camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = box
                original_label = model.names[int(class_id)]
                mapped_label = name_mapping.get(original_label, original_label)
                label = f"{mapped_label}: {conf:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/video')
def start_video():
    """Render video page"""
    return render_template('video.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process frames sent from client-side video"""
    if 'image' not in request.files:
        return 'No frame data received', 400
        
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))
    processed_image = process_image(image)
    
    # Convert processed image to bytes
    img_io = io.BytesIO()
    processed_image.save(img_io, 'JPEG')
    img_io.seek(0)
    
    return Response(img_io.getvalue(), mimetype='image/jpeg')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and detection"""
    if 'image' not in request.files:
        return 'No image uploaded', 400
    
    file = request.files['image']
    if file.filename == '':
        return 'No image selected', 400
        
    # Read and process image
    image = Image.open(io.BytesIO(file.read()))
    processed_image = process_image(image)
    
    # Save processed image to bytes
    img_io = io.BytesIO()
    processed_image.save(img_io, 'JPEG')
    img_io.seek(0)
    
    return Response(img_io.getvalue(), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
