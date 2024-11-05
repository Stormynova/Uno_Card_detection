from copy import deepcopy
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
from pathlib import Path

# Define the model path in a platform-independent way
model_path = Path("UNO_Card_Detection_YOLOv8") / "train2" / "weights" / "best.pt"
model = YOLO(str(model_path))

# Mapping for class names to more readable labels
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

# Function to process an uploaded image and annotate it with detections
def process_image(image):
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    if image_np.size == 0:
        raise ValueError("Input image is empty.")
    image_resized = cv2.resize(image_np, (640, 640), interpolation=cv2.INTER_LINEAR)

    modifications = ["original", "contrast", "brightness", "sharpen", "blur", "equalize"]
    best_results = None
    best_confidence = -1

    # Test modifications for better detection
    for modification in modifications:
        modified_img = _apply_modification(image_resized, modification)
        results = model(modified_img)
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            total_confidence = results[0].boxes.conf.cpu().numpy().mean()
            if total_confidence > best_confidence:
                best_confidence = total_confidence
                best_results = results
                best_modified_img = modified_img.copy()
                if total_confidence > 0.8:
                   break

    # Return the original if no detections found
    if best_results is None:
        return Image.fromarray(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)), [], []

    labels = []

    # Annotate detected boxes
    for result in best_results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box
            original_label = model.names[int(class_id)]
            mapped_label = name_mapping.get(original_label, original_label)  # Use mapping

            # Annotate with mapped label and confidence
            label = f"{mapped_label}: {conf:.2f}"
            labels.append(label)
            cv2.rectangle(best_modified_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(best_modified_img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    annotated_image = Image.fromarray(cv2.cvtColor(best_modified_img, cv2.COLOR_BGR2RGB))
    return annotated_image, labels, []

# Function to apply various modifications to enhance detection
def _apply_modification(image, modification):
    img = image.copy()
    if modification == "contrast":
        return cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    elif modification == "brightness":
        return cv2.convertScaleAbs(img, alpha=1.0, beta=30)
    elif modification == "sharpen":
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(img, -1, kernel)
    elif modification == "blur":
        return cv2.GaussianBlur(img, (3, 3), 0)
    elif modification == "equalize":
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img

# Function for live stream prediction with camera
def live_stream_prediction():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        results = model(frame)

        # Loop through results and annotate the frame
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()

            # Draw boxes and labels on the frame
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = box
                original_label = model.names[int(class_id)]
                mapped_label = name_mapping.get(original_label, original_label)
                label = f"{mapped_label}: {conf:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YOLO Inference", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Streamlit interface setup
st.title("UNO Card Detection")
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    if st.button("Process Image"):
        with st.spinner("Processing..."):
            processed_image, labels, _ = process_image(image)
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption='Uploaded Image', use_column_width=True)
            with col2:
                st.image(processed_image, caption='Annotated Image', use_column_width=True)
            st.subheader("Detected Objects:")
            for ind, label in enumerate(labels):
                st.write(f"{ind + 1}. {label}")

if st.button("Start Live Stream Prediction"):
    st.warning("Press 'q' to stop the live stream in the terminal.")
    live_stream_prediction()