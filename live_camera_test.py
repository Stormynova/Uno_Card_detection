# Import required libraries
import cv2  # OpenCV library for computer vision operations
from ultralytics import YOLO  # YOLO object detection model
from pathlib import Path  # For cross-platform path handling

# Define the path to the trained model weights
# Using Path ensures cross-platform compatibility (Windows/Linux/Mac)
model_path = Path("UNO_Card_Detection_YOLOv8") / "train2" / "weights" / "best.pt"
model = YOLO(str(model_path))  # Load the YOLO model with trained weights

# Dictionary mapping internal model class names to human-readable display names
# Format: "internal_name": "display_name"
# This makes the output more user-friendly by converting technical labels to readable text
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

# Initialize video capture from the default camera (index 0)
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Main loop for continuous video capture and processing
while True:
    # Capture frame-by-frame
    # ret: Boolean indicating if frame was successfully captured
    # frame: The actual image frame captured
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform object detection on the current frame
    # results contains all detected objects and their properties
    results = model(frame)

    # Process each detection result
    for result in results:
        # Extract detection information and convert to NumPy arrays
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates (x1,y1,x2,y2)
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores for each detection
        class_ids = result.boxes.cls.cpu().numpy()  # Class IDs for each detected object

        # Process each detected object
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            # Extract coordinates for bounding box
            x1, y1, x2, y2 = box
            # Get the original label from model's class names
            original_label = model.names[int(class_id)]

            # Convert technical label to human-readable format
            # If label not in mapping, keep original
            label = name_mapping.get(original_label, original_label)
            # Add confidence score to label
            label = f"{label}: {conf:.2f}"

            # Draw detection visualization on frame
            # Green rectangle around detected object
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Label text above rectangle
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the processed frame with detections
    cv2.imshow("YOLO Inference", frame)

    # Check for 'q' key press to quit
    # waitKey(1) waits 1ms for key input
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()  # Release the camera
cv2.destroyAllWindows()  # Close all OpenCV windows
