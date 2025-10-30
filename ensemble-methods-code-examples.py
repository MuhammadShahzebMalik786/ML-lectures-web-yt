"""
Ensemble Methods - Complete Code Examples for Teaching
=====================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                             GradientBoostingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DEMO 1: Bootstrap Sampling Visualization
# =============================================================================
def demo_bootstrap_sampling():
    """Demonstrate how bootstrap sampling creates diverse datasets"""
    print("=" * 50)
    print("DEMO 1: Bootstrap Sampling")
    print("=" * 50)
    
    # Original small dataset
    original = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(f"Original dataset: {original}")
    print(f"Size: {len(original)}")
    print()
    
    # Create multiple bootstrap samples
    print("Bootstrap Samples:")
    for i in range(5):
        bootstrap = np.random.choice(original, size=len(original), replace=True)
        unique_count = len(np.unique(bootstrap))
        print(f"Sample {i+1}: {bootstrap}")
        print(f"         Unique values: {unique_count}/10 ({unique_count/10*100:.0f}%)")
        print()

# =============================================================================
# DEMO 2: Single Model vs Ensemble Comparison
# =============================================================================
def demo_single_vs_ensemble():
    """Compare single model performance with ensemble"""
    print("=" * 50)
    print("DEMO 2: Single Model vs Ensemble")
    print("=" * 50)
    
    # Generate synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                              n_redundant=10, n_clusters_per_class=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Single Decision Tree
    single_tree = DecisionTreeClassifier(random_state=42)
    single_tree.fit(X_train, y_train)
    single_pred = single_tree.predict(X_test)
    single_accuracy = accuracy_score(y_test, single_pred)
    
    # Random Forest (Bagging Ensemble)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    print(f"Single Decision Tree Accuracy: {single_accuracy:.3f}")
    print(f"Random Forest Accuracy:        {rf_accuracy:.3f}")
    print(f"Improvement:                   {rf_accuracy - single_accuracy:.3f}")
    print(f"Relative Improvement:          {((rf_accuracy - single_accuracy) / single_accuracy * 100):.1f}%")

# =============================================================================
# DEMO 3: Bagging Implementation from Scratch
# =============================================================================
def demo_bagging_from_scratch():
    """Implement simple bagging from scratch"""
    print("=" * 50)
    print("DEMO 3: Bagging from Scratch")
    print("=" * 50)
    
    # Simple dataset
    X, y = make_classification(n_samples=200, n_features=10, n_informative=5,
                              n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Manual Bagging Implementation
    n_models = 10
    models = []
    predictions = []
    
    print(f"Training {n_models} models with bootstrap sampling...")
    
    for i in range(n_models):
        # Bootstrap sampling
        n_samples = len(X_train)
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_bootstrap = X_train[bootstrap_indices]
        y_bootstrap = y_train[bootstrap_indices]
        
        # Train model
        model = DecisionTreeClassifier(max_depth=5, random_state=i)
        model.fit(X_bootstrap, y_bootstrap)
        models.append(model)
        
        # Get predictions
        pred = model.predict(X_test)
        predictions.append(pred)
        
        individual_accuracy = accuracy_score(y_test, pred)
        print(f"Model {i+1} accuracy: {individual_accuracy:.3f}")
    
    # Ensemble prediction (majority voting)
    predictions = np.array(predictions)
    ensemble_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    
    print(f"\nEnsemble accuracy: {ensemble_accuracy:.3f}")
    print(f"Average individual accuracy: {np.mean([accuracy_score(y_test, pred) for pred in predictions]):.3f}")

# =============================================================================
# DEMO 4: Boosting Weight Visualization
# =============================================================================
def demo_boosting_weights():
    """Demonstrate how AdaBoost adjusts sample weights"""
    print("=" * 50)
    print("DEMO 4: AdaBoost Weight Adjustment")
    print("=" * 50)
    
    # Simple 2D dataset for visualization
    np.random.seed(42)
    X = np.random.randn(20, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # AdaBoost with few estimators to see weight changes
    ada = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=3,
        random_state=42
    )
    ada.fit(X, y)
    
    print("AdaBoost Training Results:")
    print(f"Number of estimators: {len(ada.estimators_)}")
    print(f"Estimator weights: {ada.estimator_weights_}")
    print(f"Estimator errors: {ada.estimator_errors_}")
    
    # Show how sample weights would change (conceptual)
    print("\nWeight adjustment concept:")
    print("- Correctly classified samples: weight × exp(-α)")
    print("- Incorrectly classified samples: weight × exp(+α)")
    print("- Higher α (lower error) → more dramatic weight adjustment")

# =============================================================================
# DEMO 5: Complete Ensemble Comparison
# =============================================================================
def demo_complete_ensemble_comparison():
    """Compare all ensemble methods on real dataset"""
    print("=" * 50)
    print("DEMO 5: Complete Ensemble Comparison")
    print("=" * 50)
    
    # Load real dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models
    models = {
        'Single Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest (Bagging)': RandomForestClassifier(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    
    # Add Stacking
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('ada', AdaBoostClassifier(n_estimators=50, random_state=42)),
        ('svm', SVC(probability=True, random_state=42))
    ]
    
    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(random_state=42),
        cv=5
    )
    models['Stacking'] = stacking
    
    # Evaluate all models
    print("Model Performance Comparison:")
    print("-" * 60)
    print(f"{'Model':<25} {'CV Score':<12} {'Test Score':<12} {'Std':<8}")
    print("-" * 60)
    
    results = {}
    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Test performance
        model.fit(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_score': test_score
        }
        
        print(f"{name:<25} {cv_scores.mean():.3f}        {test_score:.3f}        {cv_scores.std():.3f}")
    
    # Find best model
    best_model = max(results.keys(), key=lambda k: results[k]['test_score'])
    print("-" * 60)
    print(f"Best performing model: {best_model}")
    print(f"Test accuracy: {results[best_model]['test_score']:.3f}")

# =============================================================================
# DEMO 6: Stacking Implementation Details
# =============================================================================
def demo_stacking_details():
    """Show detailed stacking implementation"""
    print("=" * 50)
    print("DEMO 6: Stacking Implementation Details")
    print("=" * 50)
    
    # Generate data
    X, y = make_classification(n_samples=500, n_features=15, n_informative=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Base models
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('svm', SVC(probability=True, random_state=42)),
        ('nb', GaussianNB())
    ]
    
    # Meta-learner
    meta_learner = LogisticRegression(random_state=42)
    
    print("Base Models:")
    for name, model in base_models:
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        print(f"  {name.upper()}: {accuracy:.3f}")
    
    # Stacking
    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,
        passthrough=False  # Only use base model predictions
    )
    
    stacking.fit(X_train, y_train)
    stacking_accuracy = stacking.score(X_test, y_test)
    
    print(f"\nStacking Ensemble: {stacking_accuracy:.3f}")
    
    # Show meta-learner coefficients
    print(f"\nMeta-learner coefficients:")
    feature_names = [name for name, _ in base_models]
    for i, coef in enumerate(stacking.final_estimator_.coef_[0]):
        print(f"  {feature_names[i]}: {coef:.3f}")

# =============================================================================
# DEMO 7: Bias-Variance Visualization
# =============================================================================
def demo_bias_variance():
    """Demonstrate bias-variance tradeoff in ensembles"""
    print("=" * 50)
    print("DEMO 7: Bias-Variance Demonstration")
    print("=" * 50)
    
    # Generate multiple datasets to show bias-variance
    n_experiments = 50
    n_samples = 100
    
    single_predictions = []
    ensemble_predictions = []
    
    for i in range(n_experiments):
        # Generate slightly different datasets
        X, y = make_classification(n_samples=n_samples, n_features=10, 
                                  n_informative=5, random_state=i)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Single model
        single_model = DecisionTreeClassifier(random_state=42)
        single_model.fit(X_train, y_train)
        single_pred = single_model.predict(X_test)
        single_predictions.append(single_pred)
        
        # Ensemble
        ensemble_model = RandomForestClassifier(n_estimators=10, random_state=42)
        ensemble_model.fit(X_train, y_train)
        ensemble_pred = ensemble_model.predict(X_test)
        ensemble_predictions.append(ensemble_pred)
    
    # Calculate variance (disagreement across experiments)
    single_predictions = np.array(single_predictions)
    ensemble_predictions = np.array(ensemble_predictions)
    
    # Variance = average disagreement across experiments
    single_variance = np.mean(np.var(single_predictions, axis=0))
    ensemble_variance = np.mean(np.var(ensemble_predictions, axis=0))
    
    print(f"Single Model Variance: {single_variance:.3f}")
    print(f"Ensemble Variance: {ensemble_variance:.3f}")
    print(f"Variance Reduction: {((single_variance - ensemble_variance) / single_variance * 100):.1f}%")

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("ENSEMBLE METHODS - TEACHING DEMONSTRATIONS")
    print("=" * 60)
    
    # Run all demonstrations
    demo_bootstrap_sampling()
    print("\n" + "="*60 + "\n")
    
    demo_single_vs_ensemble()
    print("\n" + "="*60 + "\n")
    
    demo_bagging_from_scratch()
    print("\n" + "="*60 + "\n")
    
    demo_boosting_weights()
    print("\n" + "="*60 + "\n")
    
    demo_complete_ensemble_comparison()
    print("\n" + "="*60 + "\n")
    
    demo_stacking_details()
    print("\n" + "="*60 + "\n")
    
    demo_bias_variance()
    
    print("\n" + "="*60)
    print("ALL DEMONSTRATIONS COMPLETED")
    print("="*60)
