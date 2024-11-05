import cv2
from ultralytics import YOLO
import os
from pathlib import Path

# Define the model path in a platform-independent way
model_path = Path("UNO_Card_Detection_YOLOv8") / "train2" / "weights" / "best.pt"
model = YOLO(str(model_path))

# Mapping of original class names to desired display names
name_mapping = {
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
    "lcolor-40": "Wild",
    "lcolor-400": "Wild draw 4",
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

# Specify the image path directly
image_path = "/home/hous/Desktop/UNO/Uno_Card_detection/UNO-DATASET--3/valid/images/IMG_1911_jpg.rf.1466cd9c5427147f6037c24a982ecc14.jpg"

# Check if the image file exists
if not os.path.isfile(image_path):
    print(f"Error: File '{image_path}' does not exist.")
    exit()

# Load the image
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not read image '{image_path}'.")
    exit()

# Run inference on the image
results = model(image)

# Loop through results and annotate the image
for result in results:
    # Extract bounding boxes, confidence scores, and class IDs, converting to CPU before NumPy
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()

    # Draw boxes and labels on the image
    for box, conf, class_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = box
        original_label = model.names[int(class_id)]
        
        # Get the mapped label using the name mapping dictionary
        label = name_mapping.get(original_label, original_label)  # Fallback to original if not in mapping
        label = f"{label}: {conf:.2f}"

        # Draw bounding box and label
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save and display the result image
result_image_path = "result_image.jpg"
cv2.imwrite(result_image_path, image)
print(f"Result saved as '{result_image_path}'")

# Show the image with detections
cv2.imshow("YOLO Inference", image)
cv2.waitKey(0)
cv2.destroyAllWindows()