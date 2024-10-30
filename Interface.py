from copy import deepcopy
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# Load the YOLO model
# model = YOLO(r"./runs/detect/train/weights/best.pt")
model = YOLO(r"best.pt")

colors = {'0': 'red', '1': 'yellow', '2': 'green', '3': 'blue', '4': 'No Color'}
value = {'0': '0','1': '1','2': '2','3': '3','4': '4','5': '5','6': '6','7': '7','8': '8','9': '9','A': 'Reverse','B': 'Skip','C': 'Draw Two',}


def process_image(image):
    # Convert the image to a NumPy array for OpenCV processing
    image_np = np.array(image)

    # Convert RGB to BGR for OpenCV if necessary
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Ensure the image is not empty
    if image_np.size == 0:
        raise ValueError("Input image is empty.")

    # Resize the image as necessary for your model
    desired_size = (640, 640)  # Adjust based on your model's requirements
    image_resized = cv2.resize(image_np, desired_size, interpolation=cv2.INTER_LINEAR)

    # Get predictions from your model (assuming the model expects RGB images)
    results = model.predict(image_resized)  # Model output assumed to be in the resized size
    labels = []
    colors1 = []

    # Loop through results and annotate the resized image
    for result in results:
        boxes = result.boxes.xyxy.numpy()  # Get bounding box coordinates
        confidences = result.boxes.conf.numpy()  # Get confidence scores
        class_ids = result.boxes.cls.numpy()  # Get class IDs

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box
            label = f"{model.names[int(class_id)]}: {conf:.2f}"
            print('Label is: ', label)
            required_color = colors[label[0]]
            if '40' in label:
                label = 'Color: {} <==> Value: +4 Card'.format('No Color')
            elif '41' in label:
                label = 'Color: {} <==> Value: Wild Card'.format('No Color')
            else:
                label = 'Color: {} <==> Value: {}'.format(required_color, value[label[1]])
            labels.append(label)
            # Set the color of the box and text as needed
            cv2.rectangle(image_resized, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image_resized, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert the annotated image back to RGB format for PIL
    annotated_image = Image.fromarray(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))

    return annotated_image, labels, colors1


def live_stream_prediction():
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
            boxes = result.boxes.xyxy.numpy()  # Get bounding box coordinates
            confidences = result.boxes.conf.numpy()  # Get confidence scores
            class_ids = result.boxes.cls.numpy()  # Get class IDs

            # Draw boxes and labels on the frame
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = box
                # color = get_dominant_color(frame)
                label = f"{model.names[int(class_id)]}: {conf:.2f}"
                required_color = colors[label[0]]
                if '40' in label:
                    label = 'Color: {} <==> Value: +4 Card'.format('No Color')
                elif '41' in label:
                    label = 'Color: {} <==> Value: Wild Card'.format('No Color')
                else:
                    label = 'Color: {} <==> Value: {}'.format(required_color, value[label[1]])


                # label = f"{model.names[int(class_id)]}: {conf:.2f} :: Color: " + color
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


# Title of the app
st.title("Image Upload and Processing")

# Upload button
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    print("Path is: ", uploaded_file)

    # Button to process the image
    if st.button("Process Image"):
        with st.spinner("Processing..."):
            # Process the image
            processed_image, labels, colors = process_image(image)

            # Create two columns for side-by-side display
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption='Uploaded Image', use_column_width=True)
            with col2:
                st.image(processed_image, caption='Annotated Image', use_column_width=True)

            # Display the list of detected labels
            st.subheader("Detected Objects:")
            for ind, label in enumerate(labels):
                st.write('1. ' + label)

# Button for live stream prediction
if st.button("Start Live Stream Prediction"):
    st.warning("Press 'q' to stop the live stream in the terminal.")
    live_stream_prediction()

# Footer
st.markdown("---")
st.markdown("### About this App")
st.markdown("This app allows you to upload an image and processes it by identifying objects using YOLO.")
st.markdown("Made with ❤️ by Streamlit.")
