#!/usr/bin/env python3
"""
Data Preprocessing Examples
Complete implementation of all preprocessing techniques covered in the lecture
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """Load and explore the dataset"""
    print("="*50)
    print("1. LOADING AND EXPLORING DATA")
    print("="*50)
    
    # Load dataset
    dataset = pd.read_csv('customers.csv')
    
    print(f"Dataset shape: {dataset.shape}")
    print(f"\nFirst 5 rows:")
    print(dataset.head())
    
    print(f"\nDataset info:")
    print(dataset.info())
    
    print(f"\nMissing values:")
    print(dataset.isnull().sum())
    
    print(f"\nUnique values:")
    print(f"Countries: {dataset['Country'].unique()}")
    print(f"Purchased: {dataset['Purchased'].unique()}")
    
    return dataset

def handle_missing_data(X):
    """Handle missing data using imputation"""
    print("\n" + "="*50)
    print("2. HANDLING MISSING DATA")
    print("="*50)
    
    print("Before imputation:")
    print(f"Missing values in Age: {np.isnan(X[:, 1]).sum()}")
    print(f"Missing values in Salary: {np.isnan(X[:, 2]).sum()}")
    
    # Create imputer for numerical columns
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X[:, 1:3] = imputer.fit_transform(X[:, 1:3])
    
    print("\nAfter imputation:")
    print(f"Missing values in Age: {np.isnan(X[:, 1]).sum()}")
    print(f"Missing values in Salary: {np.isnan(X[:, 2]).sum()}")
    
    return X

def encode_categorical_data(X, y):
    """Encode categorical variables"""
    print("\n" + "="*50)
    print("3. ENCODING CATEGORICAL DATA")
    print("="*50)
    
    print("Original X shape:", X.shape)
    
    # One-hot encode independent variable (Country)
    ct = ColumnTransformer(
        transformers=[('encoder', OneHotEncoder(), [0])],
        remainder='passthrough'
    )
    X = np.array(ct.fit_transform(X))
    print("After one-hot encoding X shape:", X.shape)
    
    # Label encode dependent variable
    le = LabelEncoder()
    y = le.fit_transform(y)
    print("Encoded target values:", np.unique(y))
    
    return X, y

def split_dataset(X, y):
    """Split dataset into training and test sets"""
    print("\n" + "="*50)
    print("4. SPLITTING DATASET")
    print("="*50)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Check class distribution
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    
    print(f"Training set class distribution: {dict(zip(unique_train, counts_train))}")
    print(f"Test set class distribution: {dict(zip(unique_test, counts_test))}")
    
    return X_train, X_test, y_train, y_test

def apply_standardization(X_train, X_test):
    """Apply standardization to numerical features"""
    print("\n" + "="*50)
    print("5. STANDARDIZATION")
    print("="*50)
    
    # Create copies for standardization
    X_train_std = X_train.copy()
    X_test_std = X_test.copy()
    
    print("Before standardization:")
    print(f"Age range: {X_train_std[:, -2].min():.2f} to {X_train_std[:, -2].max():.2f}")
    print(f"Salary range: {X_train_std[:, -1].min():.2f} to {X_train_std[:, -1].max():.2f}")
    
    # Apply standardization
    sc = StandardScaler()
    X_train_std[:, -2:] = sc.fit_transform(X_train_std[:, -2:])
    X_test_std[:, -2:] = sc.transform(X_test_std[:, -2:])
    
    print("\nAfter standardization:")
    print(f"Age mean: {np.mean(X_train_std[:, -2]):.4f}, std: {np.std(X_train_std[:, -2]):.4f}")
    print(f"Salary mean: {np.mean(X_train_std[:, -1]):.4f}, std: {np.std(X_train_std[:, -1]):.4f}")
    
    return X_train_std, X_test_std, sc

def apply_normalization(X_train, X_test):
    """Apply normalization to numerical features"""
    print("\n" + "="*50)
    print("6. NORMALIZATION")
    print("="*50)
    
    # Create copies for normalization
    X_train_norm = X_train.copy()
    X_test_norm = X_test.copy()
    
    print("Before normalization:")
    print(f"Age range: {X_train_norm[:, -2].min():.2f} to {X_train_norm[:, -2].max():.2f}")
    print(f"Salary range: {X_train_norm[:, -1].min():.2f} to {X_train_norm[:, -1].max():.2f}")
    
    # Apply normalization
    scaler = MinMaxScaler()
    X_train_norm[:, -2:] = scaler.fit_transform(X_train_norm[:, -2:])
    X_test_norm[:, -2:] = scaler.transform(X_test_norm[:, -2:])
    
    print("\nAfter normalization:")
    print(f"Age range: {X_train_norm[:, -2].min():.4f} to {X_train_norm[:, -2].max():.4f}")
    print(f"Salary range: {X_train_norm[:, -1].min():.4f} to {X_train_norm[:, -1].max():.4f}")
    
    return X_train_norm, X_test_norm, scaler

def visualize_scaling_effects(X_train, X_train_std, X_train_norm):
    """Visualize the effects of different scaling methods"""
    print("\n" + "="*50)
    print("7. VISUALIZATION OF SCALING EFFECTS")
    print("="*50)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original data
    axes[0, 0].scatter(X_train[:, -2], X_train[:, -1], alpha=0.7)
    axes[0, 0].set_title('Original Data')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Salary')
    
    # Standardized data
    axes[0, 1].scatter(X_train_std[:, -2], X_train_std[:, -1], alpha=0.7, color='orange')
    axes[0, 1].set_title('Standardized Data')
    axes[0, 1].set_xlabel('Age (standardized)')
    axes[0, 1].set_ylabel('Salary (standardized)')
    
    # Normalized data
    axes[0, 2].scatter(X_train_norm[:, -2], X_train_norm[:, -1], alpha=0.7, color='green')
    axes[0, 2].set_title('Normalized Data')
    axes[0, 2].set_xlabel('Age (normalized)')
    axes[0, 2].set_ylabel('Salary (normalized)')
    
    # Histograms
    axes[1, 0].hist(X_train[:, -2], bins=10, alpha=0.7, label='Age')
    axes[1, 0].hist(X_train[:, -1]/1000, bins=10, alpha=0.7, label='Salary (k)')
    axes[1, 0].set_title('Original Distribution')
    axes[1, 0].legend()
    
    axes[1, 1].hist(X_train_std[:, -2], bins=10, alpha=0.7, label='Age')
    axes[1, 1].hist(X_train_std[:, -1], bins=10, alpha=0.7, label='Salary')
    axes[1, 1].set_title('Standardized Distribution')
    axes[1, 1].legend()
    
    axes[1, 2].hist(X_train_norm[:, -2], bins=10, alpha=0.7, label='Age')
    axes[1, 2].hist(X_train_norm[:, -1], bins=10, alpha=0.7, label='Salary')
    axes[1, 2].set_title('Normalized Distribution')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('scaling_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved as 'scaling_comparison.png'")

def complete_preprocessing_pipeline():
    """Complete preprocessing pipeline function"""
    print("\n" + "="*50)
    print("8. COMPLETE PREPROCESSING PIPELINE")
    print("="*50)
    
    def preprocess_data(file_path):
        # Load dataset
        dataset = pd.read_csv(file_path)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
        # Handle missing data
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        X[:, 1:3] = imputer.fit_transform(X[:, 1:3])
        
        # Encode categorical data
        ct = ColumnTransformer(
            transformers=[('encoder', OneHotEncoder(), [0])],
            remainder='passthrough'
        )
        X = np.array(ct.fit_transform(X))
        
        le = LabelEncoder()
        y = le.fit_transform(y)
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Feature scaling
        sc = StandardScaler()
        X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
        X_test[:, 3:] = sc.transform(X_test[:, 3:])
        
        return X_train, X_test, y_train, y_test, sc, ct, le
    
    # Apply complete pipeline
    X_train, X_test, y_train, y_test, scaler, encoder, label_encoder = preprocess_data('customers.csv')
    
    print("Complete preprocessing pipeline executed successfully!")
    print(f"Final training set shape: {X_train.shape}")
    print(f"Final test set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def main():
    """Main function to run all preprocessing examples"""
    print("DATA PREPROCESSING EXAMPLES")
    print("="*50)
    
    # 1. Load and explore data
    dataset = load_and_explore_data()
    
    # Separate features and target
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    
    # 2. Handle missing data
    X = handle_missing_data(X)
    
    # 3. Encode categorical data
    X, y = encode_categorical_data(X, y)
    
    # 4. Split dataset
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    
    # 5. Apply standardization
    X_train_std, X_test_std, std_scaler = apply_standardization(X_train, X_test)
    
    # 6. Apply normalization
    X_train_norm, X_test_norm, norm_scaler = apply_normalization(X_train, X_test)
    
    # 7. Visualize scaling effects
    visualize_scaling_effects(X_train, X_train_std, X_train_norm)
    
    # 8. Complete pipeline
    complete_preprocessing_pipeline()
    
    print("\n" + "="*50)
    print("ALL PREPROCESSING EXAMPLES COMPLETED!")
    print("="*50)

if __name__ == "__main__":
    main()
