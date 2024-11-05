# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set matplotlib style for better looking plots
plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

# Dictionary mapping the raw label names to more readable names
# Format: "internal_name": "display_name"
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

# Define paths to the dataset directories
images_dir = "UNO-DATASET--3/train/images"
labels_dir = "UNO-DATASET--3/train/labels"

# Get list of all files in both directories
images = os.listdir(images_dir)
labels = os.listdir(labels_dir)

# Example label file format:
# 13 0.784375 0.4328125 0.0390625 0.05078125
# Where: class_id x_center y_center width height

# Print dataset statistics
print(f"Number of images: {len(images)}")
print(f"Number of labels: {len(labels)}")

# Count occurrences of each class in the dataset
class_counts = {}
for label in labels:
    with open(os.path.join(labels_dir, label), 'r') as file:
        for line in file:
            # Extract class ID from each line and increment its count
            class_id = int(line.split()[0])
            class_counts[class_id] = class_counts.get(class_id, 0) + 1

# Create a mapping from numeric class IDs to human-readable names
id_to_name = {i: name for i, name in enumerate(name_mapping.values())}

# Prepare data for plotting
names = [id_to_name[class_id] for class_id in class_counts.keys()]
counts = [class_counts[class_id] for class_id in class_counts.keys()]

# Create histogram visualization
plt.figure(figsize=(15, 6))
plt.bar(names, counts)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Class Name')
plt.ylabel('Frequency')
plt.title('Histogram of UNO Card Classes')
plt.tight_layout()
plt.show()
