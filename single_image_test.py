# Import required libraries
import cv2  # OpenCV for image processing and computer vision
from ultralytics import YOLO  # YOLO object detection model
import os  # For file and path operations
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

# Define path to the test image
# This image will be used to test the model's detection capabilities
image_path = "/home/hous/Desktop/UNO/Uno_Card_detection/UNO-DATASET--3/valid/images/IMG_1911_jpg.rf.1466cd9c5427147f6037c24a982ecc14.jpg"

# Verify that the specified image file exists
# Exit the program if the file is not found to prevent runtime errors
if not os.path.isfile(image_path):
    print(f"Error: File '{image_path}' does not exist.")
    exit()

# Load the image using OpenCV
# cv2.imread returns None if the image cannot be read
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not read image '{image_path}'.")
    exit()

# Perform object detection on the image using the YOLO model
# results contains all detected objects and their properties
results = model(image)

# Process each detection result
for result in results:
    # Extract detection information and convert to NumPy arrays
    # Converting to CPU is necessary when using GPU acceleration
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

        # Draw detection visualization on image
        # Green rectangle around detected object
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # Label text above rectangle
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save the annotated image to disk
result_image_path = "result_image.jpg"
cv2.imwrite(result_image_path, image)
print(f"Result saved as '{result_image_path}'")

# Display the annotated image in a window
# Window will stay open until any key is pressed
cv2.imshow("YOLO Inference", image)
cv2.waitKey(0)  # Wait for key press
cv2.destroyAllWindows()  # Clean up display windows