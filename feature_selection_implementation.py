#!/usr/bin/env python3
"""
Feature Selection Methods Implementation
Filter Methods, Wrapper Methods, Train-Test Split & Evaluation
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    SelectKBest, f_classif, chi2, mutual_info_classif,
    RFE, RFECV, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureSelectionDemo:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def load_data(self):
        """Load and prepare dataset"""
        # Using breast cancer dataset
        data = load_breast_cancer()
        X, y = data.data, data.target
        self.feature_names = data.feature_names
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Standardize features
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        
        print(f"Dataset shape: {X.shape}")
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
    def filter_methods(self):
        """Implement Filter Methods"""
        print("\n" + "="*50)
        print("FILTER METHODS")
        print("="*50)
        
        # 1. Univariate Selection (F-test)
        print("\n1. F-test (ANOVA)")
        selector_f = SelectKBest(score_func=f_classif, k=10)
        X_train_f = selector_f.fit_transform(self.X_train, self.y_train)
        X_test_f = selector_f.transform(self.X_test)
        
        selected_features_f = self.feature_names[selector_f.get_support()]
        print(f"Selected features: {selected_features_f}")
        
        # 2. Mutual Information
        print("\n2. Mutual Information")
        selector_mi = SelectKBest(score_func=mutual_info_classif, k=10)
        X_train_mi = selector_mi.fit_transform(self.X_train, self.y_train)
        X_test_mi = selector_mi.transform(self.X_test)
        
        selected_features_mi = self.feature_names[selector_mi.get_support()]
        print(f"Selected features: {selected_features_mi}")
        
        return {
            'f_test': (X_train_f, X_test_f, selected_features_f),
            'mutual_info': (X_train_mi, X_test_mi, selected_features_mi)
        }
    
    def wrapper_methods(self):
        """Implement Wrapper Methods"""
        print("\n" + "="*50)
        print("WRAPPER METHODS")
        print("="*50)
        
        # 1. Recursive Feature Elimination (RFE)
        print("\n1. Recursive Feature Elimination (RFE)")
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        selector_rfe = RFE(estimator, n_features_to_select=10)
        X_train_rfe = selector_rfe.fit_transform(self.X_train, self.y_train)
        X_test_rfe = selector_rfe.transform(self.X_test)
        
        selected_features_rfe = self.feature_names[selector_rfe.get_support()]
        print(f"Selected features: {selected_features_rfe}")
        
        # 2. RFE with Cross-Validation
        print("\n2. RFE with Cross-Validation")
        selector_rfecv = RFECV(estimator, step=1, cv=5, scoring='accuracy')
        X_train_rfecv = selector_rfecv.fit_transform(self.X_train, self.y_train)
        X_test_rfecv = selector_rfecv.transform(self.X_test)
        
        selected_features_rfecv = self.feature_names[selector_rfecv.get_support()]
        print(f"Optimal features: {selector_rfecv.n_features_}")
        print(f"Selected features: {selected_features_rfecv}")
        
        # 3. SelectFromModel (Feature Importance)
        print("\n3. SelectFromModel (Feature Importance)")
        selector_model = SelectFromModel(estimator, threshold='median')
        X_train_model = selector_model.fit_transform(self.X_train, self.y_train)
        X_test_model = selector_model.transform(self.X_test)
        
        selected_features_model = self.feature_names[selector_model.get_support()]
        print(f"Selected features: {selected_features_model}")
        
        return {
            'rfe': (X_train_rfe, X_test_rfe, selected_features_rfe),
            'rfecv': (X_train_rfecv, X_test_rfecv, selected_features_rfecv),
            'model_based': (X_train_model, X_test_model, selected_features_model)
        }
    
    def evaluate_performance(self, X_train, X_test, method_name):
        """Evaluate model performance on training and test sets"""
        # Train multiple models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(random_state=42)
        }
        
        results = {}
        
        for model_name, model in models.items():
            # Fit model
            model.fit(X_train, self.y_train)
            
            # Predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Accuracy scores
            train_acc = accuracy_score(self.y_train, train_pred)
            test_acc = accuracy_score(self.y_test, test_pred)
            
            results[model_name] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'overfitting': train_acc - test_acc
            }
        
        print(f"\n{method_name} - Performance Results:")
        print("-" * 60)
        for model_name, metrics in results.items():
            print(f"{model_name}:")
            print(f"  Training Accuracy: {metrics['train_accuracy']:.4f}")
            print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
            print(f"  Overfitting Gap: {metrics['overfitting']:.4f}")
            print()
        
        return results
    
    def compare_all_methods(self):
        """Compare all feature selection methods"""
        print("\n" + "="*70)
        print("COMPREHENSIVE COMPARISON OF ALL METHODS")
        print("="*70)
        
        # Original data performance
        print("\nBASELINE (All Features):")
        baseline_results = self.evaluate_performance(
            self.X_train, self.X_test, "Baseline"
        )
        
        # Filter methods
        filter_results = self.filter_methods()
        
        print("\nFILTER METHODS EVALUATION:")
        f_test_results = self.evaluate_performance(
            filter_results['f_test'][0], filter_results['f_test'][1], "F-test"
        )
        
        mi_results = self.evaluate_performance(
            filter_results['mutual_info'][0], filter_results['mutual_info'][1], 
            "Mutual Information"
        )
        
        # Wrapper methods
        wrapper_results = self.wrapper_methods()
        
        print("\nWRAPPER METHODS EVALUATION:")
        rfe_results = self.evaluate_performance(
            wrapper_results['rfe'][0], wrapper_results['rfe'][1], "RFE"
        )
        
        rfecv_results = self.evaluate_performance(
            wrapper_results['rfecv'][0], wrapper_results['rfecv'][1], "RFE-CV"
        )
        
        model_results = self.evaluate_performance(
            wrapper_results['model_based'][0], wrapper_results['model_based'][1], 
            "Model-based"
        )
        
        # Summary comparison
        self.create_summary_comparison({
            'Baseline': baseline_results,
            'F-test': f_test_results,
            'Mutual Info': mi_results,
            'RFE': rfe_results,
            'RFE-CV': rfecv_results,
            'Model-based': model_results
        })
    
    def create_summary_comparison(self, all_results):
        """Create summary comparison table"""
        print("\n" + "="*80)
        print("SUMMARY COMPARISON TABLE")
        print("="*80)
        
        # Create comparison DataFrame
        comparison_data = []
        
        for method_name, method_results in all_results.items():
            for model_name, metrics in method_results.items():
                comparison_data.append({
                    'Method': method_name,
                    'Model': model_name,
                    'Train_Acc': metrics['train_accuracy'],
                    'Test_Acc': metrics['test_accuracy'],
                    'Overfitting': metrics['overfitting']
                })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Print formatted table
        print(f"{'Method':<12} {'Model':<18} {'Train Acc':<10} {'Test Acc':<10} {'Overfitting':<12}")
        print("-" * 70)
        
        for _, row in df_comparison.iterrows():
            print(f"{row['Method']:<12} {row['Model']:<18} {row['Train_Acc']:<10.4f} "
                  f"{row['Test_Acc']:<10.4f} {row['Overfitting']:<12.4f}")
        
        # Best performing method for each model
        print("\nBEST PERFORMING METHODS:")
        print("-" * 40)
        for model in df_comparison['Model'].unique():
            model_data = df_comparison[df_comparison['Model'] == model]
            best_method = model_data.loc[model_data['Test_Acc'].idxmax()]
            print(f"{model}: {best_method['Method']} (Test Acc: {best_method['Test_Acc']:.4f})")

def main():
    """Main execution function"""
    print("Feature Selection Methods - Comprehensive Implementation")
    print("=" * 60)
    
    # Initialize demo
    demo = FeatureSelectionDemo()
    
    # Load and prepare data
    demo.load_data()
    
    # Run comprehensive comparison
    demo.compare_all_methods()
    
    print("\n" + "="*60)
    print("IMPLEMENTATION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
