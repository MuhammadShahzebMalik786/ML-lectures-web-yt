#!/usr/bin/env python3
"""
Support Vector Machine (SVM) Implementation Examples
Complete practical guide with multiple real-world applications
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
from sklearn.datasets import make_classification, make_circles, load_iris, load_digits
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import pandas as pd
import seaborn as sns

class SVMExamples:
    """Comprehensive SVM examples and implementations"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def basic_classification_example(self):
        """Basic binary classification with SVM"""
        print("=== Basic SVM Classification Example ===")
        
        # Generate sample data
        X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                                 n_informative=2, random_state=42, n_clusters_per_class=1)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train SVM
        svm = SVC(kernel='linear', C=1.0, random_state=42)
        svm.fit(X_train_scaled, y_train)
        
        # Make predictions
        predictions = svm.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Number of support vectors: {svm.n_support_}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        
        return svm, X_test_scaled, y_test
    
    def kernel_comparison(self):
        """Compare different SVM kernels"""
        print("\n=== Kernel Comparison Example ===")
        
        # Create non-linearly separable data
        X, y = make_circles(n_samples=200, noise=0.1, factor=0.3, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scale the data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Test different kernels
        kernels = ['linear', 'rbf', 'poly']
        results = {}
        
        for kernel in kernels:
            svm = SVC(kernel=kernel, C=1.0, random_state=42)
            svm.fit(X_train_scaled, y_train)
            accuracy = svm.score(X_test_scaled, y_test)
            results[kernel] = accuracy
            print(f"{kernel.upper()} kernel accuracy: {accuracy:.3f}")
        
        return results
    
    def text_classification_example(self):
        """Text classification using SVM"""
        print("\n=== Text Classification Example ===")
        
        # Sample text data for sentiment analysis
        texts = [
            "I love this product, it's amazing and works perfectly!",
            "This is the worst thing I've ever bought, terrible quality",
            "Great value for money, highly recommended to everyone",
            "Completely disappointed, waste of time and money",
            "Excellent customer service and fast delivery",
            "Poor quality, broke after one day of use",
            "Outstanding performance, exceeded my expectations",
            "Horrible experience, would not recommend to anyone",
            "Perfect for my needs, exactly what I was looking for",
            "Defective product, requesting immediate refund"
        ]
        
        # Labels: 1=Positive, 0=Negative
        labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        
        # Create pipeline: Text → TF-IDF → SVM
        text_classifier = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('svm', SVC(kernel='rbf', C=1.0, random_state=42))
        ])
        
        # Train the model
        text_classifier.fit(texts, labels)
        
        # Test with new examples
        test_texts = [
            "This product is okay, nothing special",
            "Absolutely fantastic, best purchase ever!",
            "Not worth the money, very disappointing"
        ]
        
        predictions = text_classifier.predict(test_texts)
        
        print("Text Classification Results:")
        for text, pred in zip(test_texts, predictions):
            sentiment = "Positive" if pred == 1 else "Negative"
            print(f"Text: '{text}' → {sentiment}")
        
        return text_classifier
    
    def medical_diagnosis_example(self):
        """Medical diagnosis system using SVM"""
        print("\n=== Medical Diagnosis Example ===")
        
        # Simulated medical data: [age, blood_pressure, cholesterol, heart_rate, bmi]
        np.random.seed(42)
        
        # Generate synthetic medical data
        n_samples = 200
        
        # Healthy patients (younger, normal vitals)
        healthy_age = np.random.normal(40, 10, n_samples//2)
        healthy_bp = np.random.normal(120, 15, n_samples//2)
        healthy_chol = np.random.normal(180, 20, n_samples//2)
        healthy_hr = np.random.normal(70, 10, n_samples//2)
        healthy_bmi = np.random.normal(23, 3, n_samples//2)
        
        # At-risk patients (older, elevated vitals)
        risk_age = np.random.normal(60, 12, n_samples//2)
        risk_bp = np.random.normal(160, 20, n_samples//2)
        risk_chol = np.random.normal(280, 30, n_samples//2)
        risk_hr = np.random.normal(90, 15, n_samples//2)
        risk_bmi = np.random.normal(28, 4, n_samples//2)
        
        # Combine data
        X = np.column_stack([
            np.concatenate([healthy_age, risk_age]),
            np.concatenate([healthy_bp, risk_bp]),
            np.concatenate([healthy_chol, risk_chol]),
            np.concatenate([healthy_hr, risk_hr]),
            np.concatenate([healthy_bmi, risk_bmi])
        ])
        
        y = np.concatenate([np.zeros(n_samples//2), np.ones(n_samples//2)])
        
        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train SVM
        svm_medical = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        svm_medical.fit(X_train_scaled, y_train)
        
        # Evaluate
        predictions = svm_medical.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"Medical Diagnosis Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, 
                                  target_names=['Healthy', 'At-Risk']))
        
        # Test with new patient data
        new_patients = np.array([
            [45, 130, 200, 75, 24],  # Likely healthy
            [65, 170, 320, 95, 30],  # Likely at-risk
            [55, 140, 250, 80, 26]   # Borderline case
        ])
        
        new_patients_scaled = self.scaler.transform(new_patients)
        diagnoses = svm_medical.predict(new_patients_scaled)
        
        print("\nNew Patient Diagnoses:")
        for i, (patient, diagnosis) in enumerate(zip(new_patients, diagnoses)):
            status = "At-Risk" if diagnosis == 1 else "Healthy"
            print(f"Patient {i+1}: Age={patient[0]}, BP={patient[1]}, "
                  f"Chol={patient[2]}, HR={patient[3]}, BMI={patient[4]:.1f} → {status}")
        
        return svm_medical
    
    def hyperparameter_tuning_example(self):
        """Demonstrate hyperparameter tuning with GridSearchCV"""
        print("\n=== Hyperparameter Tuning Example ===")
        
        # Load iris dataset
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define parameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'linear', 'poly']
        }
        
        # Perform grid search
        svm = SVC(random_state=42)
        grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        
        # Test best model
        best_svm = grid_search.best_estimator_
        test_accuracy = best_svm.score(X_test_scaled, y_test)
        print(f"Test accuracy with best parameters: {test_accuracy:.3f}")
        
        return grid_search
    
    def svm_regression_example(self):
        """SVM for regression (SVR) example"""
        print("\n=== SVM Regression (SVR) Example ===")
        
        # Generate regression data
        np.random.seed(42)
        X = np.linspace(0, 10, 100).reshape(-1, 1)
        y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train SVR
        svr = SVR(kernel='rbf', C=1.0, gamma='scale')
        svr.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = svr.predict(X_test_scaled)
        
        # Calculate R² score
        from sklearn.metrics import r2_score
        r2 = r2_score(y_test, y_pred)
        print(f"SVR R² Score: {r2:.3f}")
        
        return svr, X_test, y_test, y_pred
    
    def image_classification_example(self):
        """Image classification using SVM on digit dataset"""
        print("\n=== Image Classification Example (Digits) ===")
        
        # Load digits dataset
        digits = load_digits()
        X, y = digits.data, digits.target
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train SVM
        svm_digits = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        svm_digits.fit(X_train_scaled, y_train)
        
        # Evaluate
        accuracy = svm_digits.score(X_test_scaled, y_test)
        print(f"Digit Classification Accuracy: {accuracy:.3f}")
        
        # Show some predictions
        predictions = svm_digits.predict(X_test_scaled[:10])
        print("\nFirst 10 predictions vs actual:")
        for i in range(10):
            print(f"Predicted: {predictions[i]}, Actual: {y_test[i]}")
        
        return svm_digits
    
    def cross_validation_example(self):
        """Demonstrate cross-validation with SVM"""
        print("\n=== Cross-Validation Example ===")
        
        # Generate data
        X, y = make_classification(n_samples=200, n_features=10, n_informative=5, 
                                 n_redundant=2, random_state=42)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform cross-validation
        svm = SVC(kernel='rbf', C=1.0, random_state=42)
        cv_scores = cross_val_score(svm, X_scaled, y, cv=5, scoring='accuracy')
        
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return cv_scores

def main():
    """Run all SVM examples"""
    print("Support Vector Machine (SVM) Implementation Examples")
    print("=" * 55)
    
    # Initialize examples class
    svm_examples = SVMExamples()
    
    # Run all examples
    try:
        # Basic classification
        svm_examples.basic_classification_example()
        
        # Kernel comparison
        svm_examples.kernel_comparison()
        
        # Text classification
        svm_examples.text_classification_example()
        
        # Medical diagnosis
        svm_examples.medical_diagnosis_example()
        
        # Hyperparameter tuning
        svm_examples.hyperparameter_tuning_example()
        
        # SVM regression
        svm_examples.svm_regression_example()
        
        # Image classification
        svm_examples.image_classification_example()
        
        # Cross-validation
        svm_examples.cross_validation_example()
        
        print("\n" + "=" * 55)
        print("All SVM examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")

if __name__ == "__main__":
    main()
