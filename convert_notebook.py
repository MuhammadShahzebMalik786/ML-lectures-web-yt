#!/usr/bin/env python3
"""
Convert Jupyter notebook to HTML for web viewing
"""

import subprocess
import os

def convert_notebook():
    try:
        # Convert notebook to HTML
        subprocess.run([
            'jupyter', 'nbconvert', 
            '--to', 'html',
            '--template', 'classic',
            'feature_selection_tutorial.ipynb'
        ], check=True)
        
        print("✅ Notebook converted to HTML successfully!")
        print("📄 View: feature_selection_tutorial.html")
        
    except subprocess.CalledProcessError:
        print("❌ Error: jupyter nbconvert not found")
        print("Install with: pip install jupyter nbconvert")
        
    except FileNotFoundError:
        print("❌ Error: Jupyter not installed")
        print("Install with: pip install jupyter")

if __name__ == "__main__":
    convert_notebook()
