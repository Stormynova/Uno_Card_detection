
# Import required libraries
from roboflow import Roboflow
import matplotlib.pyplot as plt
import ultralytics
from ultralytics import YOLO
import torch
import os
import glob

# Perform system compatibility checks for YOLO
ultralytics.checks()

# Configuration settings
training_mode = False  # Set to True to train model, False to use pre-trained weights
model_path = "UNO_Card_Detection_YOLOv8/train2/weights/"
model = YOLO(model_path + "best.pt")  # Load YOLO model
cwd = os.getcwd()  # Get current working directory

# Training block
if training_mode:
    # Initialize Roboflow and download dataset
    rf = Roboflow(api_key="HMDxCMgxf0KzCReeuRrQ")
    project = rf.workspace("unodetection").project("cartes-annotees")
    version = project.version(4)
    dataset = version.download("yolov8")
    dataset_dir = dataset.location

    # Configure device (MPS for Mac, CPU otherwise) and train model
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.train(data=dataset_dir + "/data.yaml", 
                epochs=40, 
                imgsz=800, 
                plots=True, 
                device=device)

# Visualize training results
# Plot 1: Confusion Matrix
plt.figure(1)
plt.imshow(plt.imread(cwd + "/runs/detect/train/confusion_matrix.png"))
plt.title('Confusion Matrix')

# Plot 2: Training Results
plt.figure(2) 
plt.imshow(plt.imread(cwd + "/runs/detect/train/results.png"))
plt.title('Results')

# Plot 3: Validation Batch Labels
plt.figure(3)
plt.imshow(plt.imread(cwd + "/runs/detect/train/val_batch0_labels.jpg"))
plt.title('Validation Batch Labels')
plt.show()

# Model Validation
# Validate model performance using best weights
model.val(data=cwd + "/UNO-DATASET--3/data.yaml", 
          model=cwd + "/" + model_path + "best.pt")

# Model Prediction
# Run predictions on test images
model.predict(data=cwd + "/UNO-DATASET--3/data.yaml", 
             model=cwd + "/" + model_path + "best.pt", 
             conf=0.25, 
             source=cwd + "/UNO-DATASET--3/test/images", 
             save=True)

# Visualization of Predictions
# Display first 5 predictions in a grid
plt.figure(figsize=(15, 8))
for idx, image_path in enumerate(glob.glob(cwd + "/runs/detect/predict/*.jpg")[:5]):
    plt.subplot(2, 3, idx+1)
    plt.imshow(plt.imread(image_path))
    plt.axis('off')
    plt.title(f'Prediction {idx+1}')
plt.tight_layout()
plt.show()
