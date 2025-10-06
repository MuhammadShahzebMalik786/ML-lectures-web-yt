#!/usr/bin/env python3
"""
Quick Demo: Feature Selection Methods
Run this to see all methods in action
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from feature_selection_implementation import FeatureSelectionDemo
    
    print("🚀 Starting Feature Selection Demo...")
    print("=" * 50)
    
    # Run the complete demo
    demo = FeatureSelectionDemo()
    demo.load_data()
    demo.compare_all_methods()
    
    print("\n✅ Demo completed successfully!")
    print("\nFiles created:")
    print("- feature_selection_implementation.py (Complete implementation)")
    print("- feature-selection-complete-lecture.html (Theory & examples)")
    print("- run_feature_selection_demo.py (This demo script)")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required packages:")
    print("pip install scikit-learn pandas numpy matplotlib seaborn")
    
except Exception as e:
    print(f"❌ Error running demo: {e}")
    print("Check if all required packages are installed.")
