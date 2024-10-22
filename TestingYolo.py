import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO(r".\Uno_Card_detection\runs\detect\train2\weights\best.pt")

# Initialize the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Run inference on the current frame
    results = model(frame)

    # Loop through results and annotate the frame
    for result in results:
        # Extract the bounding boxes and their scores
        boxes = result.boxes.xyxy.numpy()  # Get bounding box coordinates
        confidences = result.boxes.conf.numpy()  # Get confidence scores
        class_ids = result.boxes.cls.numpy()  # Get class IDs

        # Draw boxes and labels on the frame
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box
            label = f"{model.names[int(class_id)]}: {conf:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow("YOLO Inference", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()