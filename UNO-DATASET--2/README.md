## UNO-DATASET--2: Dynamic Scenes

### 0. Data Analysis
#### Histograms of the datasets

<img src="./img/hist.png" alt="Histogram" width="500">

#### Comparison of detection rates across different datasets:
- Mean Detection Confidence: Shows the average confidence scores with error bars for each dataset
- Average Detections per Image: Displays the mean number of cards detected in each image
- Class Distribution: Plots the distribution of detected card classes across datasets
- Total Cards Detected: Compares the total number of cards detected in each dataset
<img src="./img/compare_preds.png" alt="Class Distribution" width="500">

---
#### Definitions:
- **Confusion Matrix**: Shows how well the model classifies each UNO card type in dark conditions. The rows represent the true labels and columns represent predicted labels. Brighter colors indicate more predictions in that cell. Perfect classification would show a bright diagonal line.
- **Performance Curves**: Three key performance metrics are shown:
  - F1 Score vs Confidence Threshold: Shows how the F1 score (balance of precision and recall) changes with different confidence thresholds. The red dot marks the optimal threshold.
  - Precision vs Recall: Illustrates the tradeoff between precision (accuracy of positive predictions) and recall (ability to find all positive cases). The red dot indicates maximum precision point.
  - Recall vs Confidence Threshold: Demonstrates how recall changes with different confidence thresholds. The red dot shows the threshold with highest recall.

---
### 1. Dark Background
These images were taken in a dark room and tests the model's ability to detect cards in low-light conditions.

<img src="./img/cm_dark.png" alt="Confusion Matrix Dark" width="500">
<img src="./img/curves_dark.png" alt="Curves Dark" width="500">
<img src="./img/preds_dark.png" alt="Predictions Dark" width="500">

---
### 2. Plain Background
These images were taken in a plain background and tests the model's ability to detect cards in a simple background.

<img src="./img/cm_plain.png" alt="Confusion Matrix Plain" width="500">
<img src="./img/curves_plain.png" alt="Curves Plain" width="500">
<img src="./img/preds_plain.png" alt="Predictions Plain" width="500">

---
### 3. Non-Plain Background
These images were taken in a non-plain background and tests the model's ability to detect cards in a non-simple background. 

<img src="./img/cm_non_plain.png" alt="Confusion Matrix Non-Plain" width="500">
<img src="./img/curves_non_plain.png" alt="Curves Non-Plain" width="500">
<img src="./img/preds_non_plain.png" alt="Predictions Non-Plain" width="500">

---
### 4. Tilted Images
These images were taken with a tilted camera and tests the model's ability to detect cards in a tilted image.

<img src="./img/cm_tilted.png" alt="Confusion Matrix Tilted" width="500">
<img src="./img/curves_tilted.png" alt="Curves Tilted" width="500">
<img src="./img/preds_tilted.png" alt="Predictions Tilted" width="500">