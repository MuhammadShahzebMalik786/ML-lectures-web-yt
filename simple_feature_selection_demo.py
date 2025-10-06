#!/usr/bin/env python3
"""
Simple Feature Selection Demo
Basic implementation using only numpy and sklearn
"""

import numpy as np
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def simple_demo():
    print("🚀 Simple Feature Selection Demo")
    print("=" * 40)
    
    # Load dataset
    try:
        data = load_breast_cancer()
        X, y = data.data, data.target
        feature_names = data.feature_names
        print(f"✅ Loaded breast cancer dataset: {X.shape}")
    except:
        # Fallback to synthetic data
        X, y = make_classification(n_samples=500, n_features=20, n_informative=10, 
                                 n_redundant=5, random_state=42)
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        print(f"✅ Created synthetic dataset: {X.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    
    # 1. Baseline (all features)
    print("\n1. BASELINE (All Features)")
    print("-" * 30)
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
    
    print(f"Features: {X_train_scaled.shape[1]}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Overfitting Gap: {train_acc - test_acc:.4f}")
    
    # 2. Filter Method (F-test)
    print("\n2. FILTER METHOD (F-test)")
    print("-" * 30)
    selector_filter = SelectKBest(f_classif, k=10)
    X_train_filter = selector_filter.fit_transform(X_train_scaled, y_train)
    X_test_filter = selector_filter.transform(X_test_scaled)
    
    model.fit(X_train_filter, y_train)
    train_acc_f = accuracy_score(y_train, model.predict(X_train_filter))
    test_acc_f = accuracy_score(y_test, model.predict(X_test_filter))
    
    print(f"Features: {X_train_filter.shape[1]}")
    print(f"Train Accuracy: {train_acc_f:.4f}")
    print(f"Test Accuracy: {test_acc_f:.4f}")
    print(f"Overfitting Gap: {train_acc_f - test_acc_f:.4f}")
    
    # Show selected features
    selected_indices = selector_filter.get_support(indices=True)
    print(f"Selected features: {selected_indices[:5]}... (showing first 5)")
    
    # 3. Wrapper Method (RFE)
    print("\n3. WRAPPER METHOD (RFE)")
    print("-" * 30)
    estimator = RandomForestClassifier(n_estimators=50, random_state=42)
    selector_wrapper = RFE(estimator, n_features_to_select=10)
    X_train_wrapper = selector_wrapper.fit_transform(X_train_scaled, y_train)
    X_test_wrapper = selector_wrapper.transform(X_test_scaled)
    
    model.fit(X_train_wrapper, y_train)
    train_acc_w = accuracy_score(y_train, model.predict(X_train_wrapper))
    test_acc_w = accuracy_score(y_test, model.predict(X_test_wrapper))
    
    print(f"Features: {X_train_wrapper.shape[1]}")
    print(f"Train Accuracy: {train_acc_w:.4f}")
    print(f"Test Accuracy: {test_acc_w:.4f}")
    print(f"Overfitting Gap: {train_acc_w - test_acc_w:.4f}")
    
    # Show selected features
    selected_indices_w = selector_wrapper.get_support(indices=True)
    print(f"Selected features: {selected_indices_w[:5]}... (showing first 5)")
    
    # 4. Summary Comparison
    print("\n4. SUMMARY COMPARISON")
    print("=" * 40)
    print(f"{'Method':<15} {'Features':<10} {'Train Acc':<12} {'Test Acc':<12} {'Gap':<8}")
    print("-" * 60)
    print(f"{'Baseline':<15} {X_train_scaled.shape[1]:<10} {train_acc:<12.4f} {test_acc:<12.4f} {train_acc-test_acc:<8.4f}")
    print(f"{'Filter (F-test)':<15} {X_train_filter.shape[1]:<10} {train_acc_f:<12.4f} {test_acc_f:<12.4f} {train_acc_f-test_acc_f:<8.4f}")
    print(f"{'Wrapper (RFE)':<15} {X_train_wrapper.shape[1]:<10} {train_acc_w:<12.4f} {test_acc_w:<12.4f} {train_acc_w-test_acc_w:<8.4f}")
    
    # Best method
    methods = {
        'Baseline': test_acc,
        'Filter': test_acc_f,
        'Wrapper': test_acc_w
    }
    best_method = max(methods, key=methods.get)
    print(f"\n🏆 Best performing method: {best_method} (Test Acc: {methods[best_method]:.4f})")
    
    print("\n✅ Demo completed successfully!")
    print("\nKey Takeaways:")
    print("- Feature selection can improve or maintain performance with fewer features")
    print("- Filter methods are fast but may miss feature interactions")
    print("- Wrapper methods are slower but consider model performance")
    print("- Always evaluate on separate test data to avoid overfitting")

if __name__ == "__main__":
    simple_demo()
