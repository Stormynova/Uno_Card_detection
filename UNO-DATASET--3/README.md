# Model Performance Visualization on UNO-DATASET--3

## Predictions

<img src="./img/preds.png" alt="Prediction" width="500">

## Confusion Matrices
The confusion matrix shows how well our model classifies each type of UNO card. The brighter squares along the diagonal indicate accurate predictions.
<img src="./img/confusion_matrix.png" alt="Confusion Matrix" width="500">

The normalized confusion matrix provides the same information but as percentages, making it easier to interpret the model's performance across classes.
<img src="./img/confusion_matrix_normalized.png" alt="Confusion Matrix Normalized" width="500">

## Performance Metrics

The F1 curve shows the balance between precision and recall across different confidence thresholds. Higher curves indicate better overall performance.
<img src="./img/F1_curve.png" alt="Precision, Recall, F1-Score, Support" width="500">

The Precision-Recall curve demonstrates the trade-off between precision and recall. The closer the curve is to the top-right corner, the better the model performs.
<img src="./img/PR_curve.png" alt="Precision-Recall Curve" width="500">

The Recall curve shows how well the model identifies positive cases at different confidence thresholds. A higher curve indicates better recall performance.
<img src="./img/R_curve.png" alt="Recall Curve" width="500">