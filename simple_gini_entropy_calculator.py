#!/usr/bin/env python3
"""
Simple Gini Impurity & Entropy Calculator
No external dependencies - pure Python implementation
"""

import math

def gini_impurity(class_counts):
    """Calculate Gini impurity from class counts"""
    total = sum(class_counts)
    if total == 0:
        return 0
    
    gini = 1.0
    for count in class_counts:
        probability = count / total
        gini -= probability ** 2
    
    return gini

def entropy(class_counts):
    """Calculate entropy from class counts"""
    total = sum(class_counts)
    if total == 0:
        return 0
    
    entropy_val = 0.0
    for count in class_counts:
        if count > 0:
            probability = count / total
            entropy_val -= probability * math.log2(probability)
    
    return entropy_val

def information_gain(parent_counts, left_counts, right_counts, criterion='gini'):
    """Calculate information gain from a split"""
    total_parent = sum(parent_counts)
    total_left = sum(left_counts)
    total_right = sum(right_counts)
    
    if criterion == 'gini':
        parent_impurity = gini_impurity(parent_counts)
        left_impurity = gini_impurity(left_counts)
        right_impurity = gini_impurity(right_counts)
    else:  # entropy
        parent_impurity = entropy(parent_counts)
        left_impurity = entropy(left_counts)
        right_impurity = entropy(right_counts)
    
    # Weighted average of child impurities
    weighted_impurity = (total_left/total_parent) * left_impurity + (total_right/total_parent) * right_impurity
    
    return parent_impurity - weighted_impurity

def print_analysis(class_counts, label="Dataset"):
    """Print detailed analysis of a dataset"""
    total = sum(class_counts)
    probabilities = [count/total for count in class_counts]
    
    gini_val = gini_impurity(class_counts)
    entropy_val = entropy(class_counts)
    
    print(f"\n📊 {label} Analysis:")
    print(f"   Class counts: {class_counts}")
    print(f"   Total samples: {total}")
    print(f"   Probabilities: {[f'{p:.3f}' for p in probabilities]}")
    print(f"   Gini Impurity: {gini_val:.4f}")
    print(f"   Entropy: {entropy_val:.4f}")
    
    # Interpretation
    if gini_val < 0.1:
        print("   🎯 Very pure dataset!")
    elif gini_val < 0.3:
        print("   ✅ Relatively pure dataset")
    elif gini_val < 0.4:
        print("   ⚠️ Moderate impurity")
    else:
        print("   🔄 High impurity - very mixed classes")

def demonstrate_examples():
    """Show various examples with different class distributions"""
    print("🌳 GINI IMPURITY & ENTROPY EXAMPLES")
    print("=" * 50)
    
    examples = [
        ([10, 0], "Pure Dataset (All Class A)"),
        ([8, 2], "Mostly Pure (80% A, 20% B)"),
        ([7, 3], "Moderately Pure (70% A, 30% B)"),
        ([6, 4], "Balanced-ish (60% A, 40% B)"),
        ([5, 5], "Perfectly Balanced (50% A, 50% B)"),
        ([3, 3, 3], "Three Classes (Equal)"),
        ([6, 2, 2], "Three Classes (Imbalanced)"),
    ]
    
    for counts, description in examples:
        print_analysis(counts, description)

def demonstrate_information_gain():
    """Demonstrate information gain calculation"""
    print("\n" + "=" * 60)
    print("🎯 INFORMATION GAIN DEMONSTRATION")
    print("=" * 60)
    
    # Original dataset: 60% class A, 40% class B
    parent = [6, 4]
    print_analysis(parent, "Original Dataset")
    
    # Split 1: Good split
    left1 = [5, 1]   # Left branch: mostly A
    right1 = [1, 3]  # Right branch: mostly B
    
    print(f"\n🌿 Split 1 (Good Split):")
    print_analysis(left1, "Left Branch")
    print_analysis(right1, "Right Branch")
    
    gini_gain1 = information_gain(parent, left1, right1, 'gini')
    entropy_gain1 = information_gain(parent, left1, right1, 'entropy')
    
    print(f"\n📈 Information Gain:")
    print(f"   Gini Gain: {gini_gain1:.4f}")
    print(f"   Entropy Gain: {entropy_gain1:.4f}")
    
    # Split 2: Poor split
    left2 = [3, 2]   # Left branch: mixed
    right2 = [3, 2]  # Right branch: mixed
    
    print(f"\n🌿 Split 2 (Poor Split):")
    print_analysis(left2, "Left Branch")
    print_analysis(right2, "Right Branch")
    
    gini_gain2 = information_gain(parent, left2, right2, 'gini')
    entropy_gain2 = information_gain(parent, left2, right2, 'entropy')
    
    print(f"\n📈 Information Gain:")
    print(f"   Gini Gain: {gini_gain2:.4f}")
    print(f"   Entropy Gain: {entropy_gain2:.4f}")
    
    print(f"\n💡 Conclusion:")
    print(f"   Split 1 is better (higher information gain)")
    print(f"   Good splits separate classes more effectively")

def interactive_calculator():
    """Interactive calculator for custom inputs"""
    print("\n" + "=" * 60)
    print("🧮 INTERACTIVE CALCULATOR")
    print("=" * 60)
    print("Enter class counts separated by spaces (e.g., '6 4' or '3 3 4')")
    print("Type 'quit' to exit")
    
    while True:
        try:
            user_input = input("\nClass counts: ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            counts = list(map(int, user_input.split()))
            
            if len(counts) < 2:
                print("❌ Please enter at least 2 class counts")
                continue
            
            if any(count < 0 for count in counts):
                print("❌ Class counts must be non-negative")
                continue
            
            print_analysis(counts, "Your Dataset")
            
            # Ask for split if they want
            print("\nWould you like to test a split? (y/n)")
            if input().lower().startswith('y'):
                print("Enter left branch counts:")
                left_counts = list(map(int, input("Left: ").split()))
                print("Enter right branch counts:")
                right_counts = list(map(int, input("Right: ").split()))
                
                if len(left_counts) == len(counts) and len(right_counts) == len(counts):
                    gini_gain = information_gain(counts, left_counts, right_counts, 'gini')
                    entropy_gain = information_gain(counts, left_counts, right_counts, 'entropy')
                    
                    print(f"\n📈 Information Gain from Split:")
                    print(f"   Gini Gain: {gini_gain:.4f}")
                    print(f"   Entropy Gain: {entropy_gain:.4f}")
                else:
                    print("❌ Split counts must have same number of classes as parent")
            
        except ValueError:
            print("❌ Invalid input. Please enter space-separated integers.")
        except Exception as e:
            print(f"❌ Error: {e}")

def comparison_table():
    """Show comparison between Gini and Entropy"""
    print("\n" + "=" * 60)
    print("⚖️ GINI vs ENTROPY COMPARISON")
    print("=" * 60)
    
    scenarios = [
        ([10, 0], "Pure (100:0)"),
        ([9, 1], "Very Pure (90:10)"),
        ([8, 2], "Mostly Pure (80:20)"),
        ([7, 3], "Moderate (70:30)"),
        ([6, 4], "Balanced-ish (60:40)"),
        ([5, 5], "Perfect Balance (50:50)"),
    ]
    
    print(f"{'Scenario':<20} {'Gini':<8} {'Entropy':<8} {'Difference':<10}")
    print("-" * 50)
    
    for counts, description in scenarios:
        gini_val = gini_impurity(counts)
        entropy_val = entropy(counts)
        diff = abs(entropy_val - gini_val)
        
        print(f"{description:<20} {gini_val:<8.3f} {entropy_val:<8.3f} {diff:<10.3f}")

def main():
    """Main function to run all demonstrations"""
    print("🌳 GINI IMPURITY & ENTROPY CALCULATOR")
    print("Pure Python Implementation - No Dependencies Required")
    print("=" * 60)
    
    # Show examples
    demonstrate_examples()
    
    # Show information gain
    demonstrate_information_gain()
    
    # Show comparison table
    comparison_table()
    
    # Key insights
    print("\n" + "=" * 60)
    print("🔑 KEY INSIGHTS")
    print("=" * 60)
    print("1. Both measures range from 0 (pure) to maximum (most mixed)")
    print("2. Gini: 0 to 0.5 (binary), Entropy: 0 to 1 (binary)")
    print("3. Gini is faster (no logarithms), Entropy more sensitive")
    print("4. Higher information gain = better split")
    print("5. Both lead to similar decision trees in practice")
    
    # Interactive section
    interactive_calculator()
    
    print("\n🎓 Thanks for learning about Gini and Entropy!")
    print("These concepts are fundamental to decision trees and random forests.")

if __name__ == "__main__":
    main()
