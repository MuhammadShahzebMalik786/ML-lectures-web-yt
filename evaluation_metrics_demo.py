#!/usr/bin/env python3
"""
Machine Learning Evaluation Metrics - Complete Demo
Author: ML Course
Date: 2025-10-03
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, r2_score,
    mean_squared_error, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_classification, make_regression

def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)

def demo_confusion_matrix():
    """Demonstrate confusion matrix and basic metrics"""
    print_section("1. CONFUSION MATRIX & BASIC METRICS")
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print("Confusion Matrix:")
    print(cm)
    
    # Extract values
    TP = cm[1,1]  # True Positives
    TN = cm[0,0]  # True Negatives
    FP = cm[0,1]  # False Positives
    FN = cm[1,0]  # False Negatives
    
    print(f"\nConfusion Matrix Components:")
    print(f"True Positives (TP):  {TP}")
    print(f"True Negatives (TN):  {TN}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    
    return y_test, y_pred, cm, TP, TN, FP, FN

def demo_accuracy(y_test, y_pred, TP, TN, FP, FN):
    """Demonstrate accuracy calculation"""
    print_section("2. ACCURACY")
    
    # Using sklearn
    accuracy = accuracy_score(y_test, y_pred)
    
    # Manual calculation
    manual_accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    print(f"Accuracy Formula: (TP + TN) / (TP + TN + FP + FN)")
    print(f"Sklearn Accuracy: {accuracy:.4f}")
    print(f"Manual Accuracy:  {manual_accuracy:.4f}")
    print(f"Percentage:       {accuracy*100:.2f}%")

def demo_precision(y_test, y_pred, TP, FP):
    """Demonstrate precision calculation"""
    print_section("3. PRECISION")
    
    # Using sklearn
    precision = precision_score(y_test, y_pred)
    
    # Manual calculation
    manual_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    print(f"Precision Formula: TP / (TP + FP)")
    print(f"Sklearn Precision: {precision:.4f}")
    print(f"Manual Precision:  {manual_precision:.4f}")
    print(f"Interpretation: Of all positive predictions, {precision*100:.1f}% were correct")
    
    return precision

def demo_recall(y_test, y_pred, TP, FN):
    """Demonstrate recall calculation"""
    print_section("4. RECALL (SENSITIVITY)")
    
    # Using sklearn
    recall = recall_score(y_test, y_pred)
    
    # Manual calculation
    manual_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    print(f"Recall Formula: TP / (TP + FN)")
    print(f"Sklearn Recall: {recall:.4f}")
    print(f"Manual Recall:  {manual_recall:.4f}")
    print(f"Interpretation: Found {recall*100:.1f}% of all actual positive cases")
    
    return recall

def demo_f1_score(y_test, y_pred, precision, recall):
    """Demonstrate F1-score calculation"""
    print_section("5. F1-SCORE")
    
    # Using sklearn
    f1 = f1_score(y_test, y_pred)
    
    # Manual calculation
    manual_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"F1-Score Formula: 2 × (Precision × Recall) / (Precision + Recall)")
    print(f"Sklearn F1-Score: {f1:.4f}")
    print(f"Manual F1-Score:  {manual_f1:.4f}")
    print(f"Interpretation: Harmonic mean balancing precision and recall")

def demo_r2_score():
    """Demonstrate R² score for regression"""
    print_section("6. R² SCORE (REGRESSION)")
    
    # Generate regression data
    X_reg, y_reg = make_regression(n_samples=1000, n_features=1, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)
    
    # Train regression model
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)
    y_pred = reg_model.predict(X_test)
    
    # Calculate R² score
    r2 = r2_score(y_test, y_pred)
    
    # Manual calculation
    ss_res = np.sum((y_test - y_pred) ** 2)  # Sum of squares of residuals
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)  # Total sum of squares
    manual_r2 = 1 - (ss_res / ss_tot)
    
    print(f"R² Formula: 1 - (SS_res / SS_tot)")
    print(f"Sklearn R²: {r2:.4f}")
    print(f"Manual R²:  {manual_r2:.4f}")
    print(f"Interpretation: Model explains {r2*100:.1f}% of the variance")
    
    # Additional regression metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"\nAdditional Regression Metrics:")
    print(f"Mean Squared Error (MSE):  {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

def demo_classification_report(y_test, y_pred):
    """Show complete classification report"""
    print_section("7. COMPLETE CLASSIFICATION REPORT")
    
    report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])
    print(report)

def interactive_calculator():
    """Interactive confusion matrix calculator"""
    print_section("8. INTERACTIVE CALCULATOR")
    
    print("Enter confusion matrix values:")
    try:
        tp = int(input("True Positives (TP): ") or "85")
        fp = int(input("False Positives (FP): ") or "15")
        tn = int(input("True Negatives (TN): ") or "90")
        fn = int(input("False Negatives (FN): ") or "10")
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nCalculated Metrics:")
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
        
    except (ValueError, KeyboardInterrupt):
        print("Using default values...")
        tp, fp, tn, fn = 85, 15, 90, 10
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        
        print(f"Default Metrics (TP={tp}, FP={fp}, TN={tn}, FN={fn}):")
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")

def main():
    """Main function to run all demonstrations"""
    print("MACHINE LEARNING EVALUATION METRICS - COMPLETE DEMO")
    print("=" * 60)
    
    # Run all demonstrations
    y_test, y_pred, cm, TP, TN, FP, FN = demo_confusion_matrix()
    demo_accuracy(y_test, y_pred, TP, TN, FP, FN)
    precision = demo_precision(y_test, y_pred, TP, FP)
    recall = demo_recall(y_test, y_pred, TP, FN)
    demo_f1_score(y_test, y_pred, precision, recall)
    demo_r2_score()
    demo_classification_report(y_test, y_pred)
    interactive_calculator()
    
    print_section("SUMMARY")
    print("Key Takeaways:")
    print("• Accuracy: Overall correctness")
    print("• Precision: Quality of positive predictions")
    print("• Recall: Coverage of actual positives")
    print("• F1-Score: Balance between precision and recall")
    print("• R²: Variance explained in regression")
    print("• Choose metrics based on your problem context!")

if __name__ == "__main__":
    main()
