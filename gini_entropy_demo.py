#!/usr/bin/env python3
"""
Gini Impurity & Entropy Demonstration
Interactive visualizations and examples for understanding decision tree splitting criteria
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def gini_impurity(y):
    """Calculate Gini impurity for a given set of labels"""
    if len(y) == 0:
        return 0
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1 - np.sum(probabilities**2)

def entropy(y):
    """Calculate entropy for a given set of labels"""
    if len(y) == 0:
        return 0
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    # Add small epsilon to avoid log(0)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

def information_gain(parent, left_child, right_child, criterion='gini'):
    """Calculate information gain from a split"""
    if criterion == 'gini':
        parent_impurity = gini_impurity(parent)
        left_impurity = gini_impurity(left_child)
        right_impurity = gini_impurity(right_child)
    else:  # entropy
        parent_impurity = entropy(parent)
        left_impurity = entropy(left_child)
        right_impurity = entropy(right_child)
    
    n_parent = len(parent)
    n_left = len(left_child)
    n_right = len(right_child)
    
    weighted_impurity = (n_left/n_parent) * left_impurity + (n_right/n_parent) * right_impurity
    return parent_impurity - weighted_impurity

def plot_impurity_comparison():
    """Plot Gini vs Entropy comparison"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Gini vs Entropy curves
    p_values = np.linspace(0.001, 0.999, 1000)
    gini_values = 2 * p_values * (1 - p_values)  # Binary Gini
    entropy_values = -(p_values * np.log2(p_values) + (1-p_values) * np.log2(1-p_values))
    
    ax1.plot(p_values, gini_values, 'r-', linewidth=3, label='Gini Impurity')
    ax1.plot(p_values, entropy_values, 'b-', linewidth=3, label='Entropy')
    ax1.set_xlabel('Probability of Class 1')
    ax1.set_ylabel('Impurity Measure')
    ax1.set_title('Gini vs Entropy Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Different class distributions
    scenarios = ['Pure\n(100:0)', '90:10', '80:20', '70:30', '60:40', '50:50']
    gini_vals = [0, 0.18, 0.32, 0.42, 0.48, 0.5]
    entropy_vals = [0, 0.47, 0.72, 0.88, 0.97, 1.0]
    
    x_pos = np.arange(len(scenarios))
    width = 0.35
    
    ax2.bar(x_pos - width/2, gini_vals, width, label='Gini', color='red', alpha=0.7)
    ax2.bar(x_pos + width/2, entropy_vals, width, label='Entropy', color='blue', alpha=0.7)
    ax2.set_xlabel('Class Distribution (A:B)')
    ax2.set_ylabel('Impurity Value')
    ax2.set_title('Impurity for Different Distributions')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(scenarios)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Information Gain Example
    # Original dataset: 60% class A, 40% class B
    original_gini = gini_impurity([0]*6 + [1]*4)
    original_entropy = entropy([0]*6 + [1]*4)
    
    # After split: Left (80% A, 20% B), Right (20% A, 80% B)
    left_split = [0]*4 + [1]*1  # 5 samples
    right_split = [0]*2 + [1]*3  # 5 samples
    
    left_gini = gini_impurity(left_split)
    right_gini = gini_impurity(right_split)
    weighted_gini = 0.5 * left_gini + 0.5 * right_gini
    
    left_entropy = entropy(left_split)
    right_entropy = entropy(right_split)
    weighted_entropy = 0.5 * left_entropy + 0.5 * right_entropy
    
    categories = ['Original', 'After Split\n(Weighted)']
    gini_comparison = [original_gini, weighted_gini]
    entropy_comparison = [original_entropy, weighted_entropy]
    
    x_pos = np.arange(len(categories))
    ax3.bar(x_pos - width/2, gini_comparison, width, label='Gini', color='red', alpha=0.7)
    ax3.bar(x_pos + width/2, entropy_comparison, width, label='Entropy', color='blue', alpha=0.7)
    ax3.set_ylabel('Impurity Value')
    ax3.set_title('Information Gain Visualization')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(categories)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add information gain annotations
    gini_gain = original_gini - weighted_gini
    entropy_gain = original_entropy - weighted_entropy
    ax3.text(0.5, max(gini_comparison + entropy_comparison) * 0.8, 
             f'Gini Gain: {gini_gain:.3f}\nEntropy Gain: {entropy_gain:.3f}',
             ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 4. Sensitivity Analysis
    base_prob = 0.5
    perturbations = np.linspace(-0.1, 0.1, 21)
    gini_sensitivity = []
    entropy_sensitivity = []
    
    for delta in perturbations:
        p = base_prob + delta
        if 0 < p < 1:
            gini_val = 2 * p * (1 - p)
            entropy_val = -(p * np.log2(p) + (1-p) * np.log2(1-p))
        else:
            gini_val = 0
            entropy_val = 0
        gini_sensitivity.append(gini_val)
        entropy_sensitivity.append(entropy_val)
    
    ax4.plot(perturbations, gini_sensitivity, 'r-', linewidth=3, label='Gini')
    ax4.plot(perturbations, entropy_sensitivity, 'b-', linewidth=3, label='Entropy')
    ax4.set_xlabel('Probability Change from 0.5')
    ax4.set_ylabel('Impurity Value')
    ax4.set_title('Sensitivity to Probability Changes')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def demonstrate_decision_tree():
    """Demonstrate decision tree with different criteria"""
    # Generate sample data
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                              n_informative=2, n_clusters_per_class=1, 
                              random_state=42)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Decision tree with Gini
    dt_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
    dt_gini.fit(X, y)
    
    # Decision tree with Entropy
    dt_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
    dt_entropy.fit(X, y)
    
    # Plot trees
    plot_tree(dt_gini, ax=ax1, filled=True, feature_names=['Feature 1', 'Feature 2'], 
              class_names=['Class 0', 'Class 1'], fontsize=10)
    ax1.set_title('Decision Tree with Gini Criterion')
    
    plot_tree(dt_entropy, ax=ax2, filled=True, feature_names=['Feature 1', 'Feature 2'], 
              class_names=['Class 0', 'Class 1'], fontsize=10)
    ax2.set_title('Decision Tree with Entropy Criterion')
    
    plt.tight_layout()
    plt.show()
    
    return dt_gini, dt_entropy

def interactive_example():
    """Interactive example with user input"""
    print("\n" + "="*60)
    print("🌳 INTERACTIVE GINI & ENTROPY CALCULATOR")
    print("="*60)
    
    while True:
        try:
            print("\nEnter class counts (e.g., '6 4' for 6 of class A, 4 of class B):")
            print("Or type 'quit' to exit")
            
            user_input = input("Class counts: ").strip()
            
            if user_input.lower() == 'quit':
                break
                
            counts = list(map(int, user_input.split()))
            
            if len(counts) < 2:
                print("Please enter at least 2 class counts")
                continue
                
            # Create label array
            labels = []
            for i, count in enumerate(counts):
                labels.extend([i] * count)
            
            # Calculate metrics
            gini_val = gini_impurity(labels)
            entropy_val = entropy(labels)
            
            total_samples = sum(counts)
            probabilities = [count/total_samples for count in counts]
            
            print(f"\n📊 Results for {counts}:")
            print(f"Total samples: {total_samples}")
            print(f"Class probabilities: {[f'{p:.3f}' for p in probabilities]}")
            print(f"Gini Impurity: {gini_val:.4f}")
            print(f"Entropy: {entropy_val:.4f}")
            
            # Interpretation
            if gini_val < 0.1:
                print("🎯 Very pure dataset!")
            elif gini_val < 0.3:
                print("✅ Relatively pure dataset")
            elif gini_val < 0.4:
                print("⚠️ Moderate impurity")
            else:
                print("🔄 High impurity - very mixed classes")
                
        except ValueError:
            print("❌ Invalid input. Please enter space-separated integers.")
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main demonstration function"""
    print("🌳 GINI IMPURITY & ENTROPY DEMONSTRATION")
    print("="*50)
    
    # Example calculations
    print("\n1. 📊 Example Calculations:")
    example_data = [0, 0, 0, 1, 1, 1, 1, 0, 0, 1]  # 5 class 0, 5 class 1
    print(f"Dataset: {example_data}")
    print(f"Gini Impurity: {gini_impurity(example_data):.4f}")
    print(f"Entropy: {entropy(example_data):.4f}")
    
    # Information gain example
    print("\n2. 🎯 Information Gain Example:")
    parent = [0, 0, 0, 1, 1, 1, 1, 0, 0, 1]
    left_child = [0, 0, 0, 1]
    right_child = [1, 1, 1, 0, 0, 1]
    
    gini_gain = information_gain(parent, left_child, right_child, 'gini')
    entropy_gain = information_gain(parent, left_child, right_child, 'entropy')
    
    print(f"Parent: {parent}")
    print(f"Left child: {left_child}")
    print(f"Right child: {right_child}")
    print(f"Gini Information Gain: {gini_gain:.4f}")
    print(f"Entropy Information Gain: {entropy_gain:.4f}")
    
    # Visualizations
    print("\n3. 📈 Generating Visualizations...")
    plot_impurity_comparison()
    
    print("\n4. 🌳 Decision Tree Comparison...")
    dt_gini, dt_entropy = demonstrate_decision_tree()
    
    # Performance comparison
    print(f"\n5. ⚡ Performance Comparison:")
    print(f"Gini Tree Accuracy: {dt_gini.score(*make_classification(n_samples=100, n_features=2, random_state=42)[:2]):.3f}")
    print(f"Entropy Tree Accuracy: {dt_entropy.score(*make_classification(n_samples=100, n_features=2, random_state=42)[:2]):.3f}")
    
    # Interactive section
    interactive_example()

if __name__ == "__main__":
    main()
