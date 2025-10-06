# Machine Learning Evaluation Metrics - Complete Guide

This lecture covers all essential evaluation metrics for machine learning models with interactive examples and code demonstrations.

## Files Created

1. **`evaluation-metrics-lecture.html`** - Interactive web-based lecture
2. **`evaluation_metrics_demo.py`** - Python script with complete demonstrations
3. **`EVALUATION_METRICS_README.md`** - This documentation file

## How to Use

### Web Version (HTML)
1. Open `evaluation-metrics-lecture.html` in any web browser
2. Click "Run Code" buttons to see example outputs
3. Use the interactive calculator at the bottom
4. All code examples are displayed with syntax highlighting

### Python Script Version
```bash
# Make sure you have required packages
pip install numpy scikit-learn matplotlib seaborn

# Run the complete demonstration
python evaluation_metrics_demo.py
```

## Topics Covered

### 1. Confusion Matrix
- True Positives (TP), True Negatives (TN)
- False Positives (FP), False Negatives (FN)
- Visual representation and interpretation

### 2. Classification Metrics
- **Accuracy**: Overall correctness `(TP + TN) / Total`
- **Precision**: Quality of positive predictions `TP / (TP + FP)`
- **Recall**: Coverage of actual positives `TP / (TP + FN)`
- **F1-Score**: Harmonic mean of precision and recall

### 3. Regression Metrics
- **R² Score**: Coefficient of determination
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error

### 4. Interactive Features
- Live metric calculator
- Code execution simulation
- Visual confusion matrix
- Formula explanations

## Key Formulas

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
R² = 1 - (SS_res / SS_tot)
```

## When to Use Each Metric

- **Accuracy**: Balanced datasets, overall performance
- **Precision**: When false positives are costly (spam detection)
- **Recall**: When false negatives are costly (medical diagnosis)
- **F1-Score**: Imbalanced datasets, need balance between precision/recall
- **R²**: Regression problems, variance explanation

## Requirements

- Python 3.6+
- scikit-learn
- numpy
- matplotlib (for Python version)
- seaborn (for Python version)
- Modern web browser (for HTML version)

## Features

✅ Interactive web interface
✅ Executable Python demonstrations
✅ Manual calculation examples
✅ Real dataset examples
✅ Visual confusion matrix
✅ Complete classification reports
✅ Regression metrics included
✅ Interactive calculator

The lecture follows the same template structure as other ML lectures in this directory and provides both theoretical understanding and practical implementation of evaluation metrics.
