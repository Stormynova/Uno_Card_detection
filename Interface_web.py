# Import necessary libraries
from copy import deepcopy  # For creating deep copies of objects
import streamlit as st    # For creating web interface
from PIL import Image     # For image processing
import numpy as np        # For numerical operations
import cv2               # For computer vision operations
from ultralytics import YOLO  # For YOLO object detection
from pathlib import Path      # For handling file paths cross-platform

# Define the model path in a platform-independent way using Path
# This ensures the path works on both Windows and Unix systems
model_path = Path("UNO_Card_Detection_YOLOv8") / "train2" / "weights" / "best.pt"
model = YOLO(str(model_path))  # Load the YOLO model

# Dictionary mapping internal class names to human-readable labels
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

def process_image(image):
    """
    Process an uploaded image and detect UNO cards within it
    Args:
        image: PIL Image object to process
    Returns:
        tuple: (annotated image, list of detected labels, empty list)
    """
    # Convert PIL image to numpy array and change color space
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    if image_np.size == 0:
        raise ValueError("Input image is empty.")
    # Resize image to model's expected input size
    image_resized = cv2.resize(image_np, (640, 640), interpolation=cv2.INTER_LINEAR)

    # List of possible image modifications to try
    modifications = ["original", "contrast", "brightness", "sharpen", "blur", "equalize"]
    best_results = None
    best_confidence = -1

    # Try different image modifications to get best detection results
    for modification in modifications:
        modified_img = _apply_modification(image_resized, modification)
        results = model(modified_img)
        
        # If detections found, check if they're better than previous best
        if len(results) > 0 and len(results[0].boxes) > 0:
            total_confidence = results[0].boxes.conf.cpu().numpy().mean()
            if total_confidence > best_confidence:
                best_confidence = total_confidence
                best_results = results
                best_modified_img = modified_img.copy()
                # If confidence is very high, no need to try more modifications
                if total_confidence > 0.8:
                   break

    # If no detections found, return original image
    if best_results is None:
        return Image.fromarray(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)), [], []

    labels = []

    # Draw boxes and labels for each detected card
    for result in best_results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box
            original_label = model.names[int(class_id)]
            mapped_label = name_mapping.get(original_label, original_label)

            # Create label with confidence score
            label = f"{mapped_label}: {conf:.2f}"
            labels.append(label)
            # Draw rectangle around detected card
            cv2.rectangle(best_modified_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Add text label above rectangle
            cv2.putText(best_modified_img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert back to PIL Image for display
    annotated_image = Image.fromarray(cv2.cvtColor(best_modified_img, cv2.COLOR_BGR2RGB))
    return annotated_image, labels, []

def _apply_modification(image, modification):
    """
    Apply various image modifications to potentially improve detection
    Args:
        image: numpy array of image
        modification: string indicating type of modification
    Returns:
        modified image as numpy array
    """
    img = image.copy()
    if modification == "contrast":
        return cv2.convertScaleAbs(img, alpha=1.5, beta=0)  # Increase contrast
    elif modification == "brightness":
        return cv2.convertScaleAbs(img, alpha=1.0, beta=30)  # Increase brightness
    elif modification == "sharpen":
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # Sharpening kernel
        return cv2.filter2D(img, -1, kernel)
    elif modification == "blur":
        return cv2.GaussianBlur(img, (3, 3), 0)  # Apply Gaussian blur
    elif modification == "equalize":
        # Equalize histogram in YUV color space
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img  # Return original if no modification specified

# Set up the Streamlit web interface
st.title("UNO Card Detection")
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

# Handle image upload and processing
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    if st.button("Process Image"):
        with st.spinner("Processing..."):
            # Process the image and display results
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
    st.warning("Live Stream Prediction is not available in the web version.")