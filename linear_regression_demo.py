#!/usr/bin/env python3
"""
Linear Regression Demo - 3 Minute ML Lecture
Simple examples to understand linear regression
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def basic_example():
    """Basic linear regression example"""
    print("=== Basic Example: Study Hours vs Exam Score ===")
    
    # Data
    X = np.array([[1], [2], [3], [4], [5], [6]])  # Hours studied
    y = np.array([50, 60, 70, 80, 90, 95])        # Exam scores
    
    # Model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predictions
    predictions = model.predict(X)
    new_prediction = model.predict([[7]])
    
    print(f"Slope (m): {model.coef_[0]:.2f}")
    print(f"Intercept (b): {model.intercept_:.2f}")
    print(f"Equation: Score = {model.coef_[0]:.2f} * Hours + {model.intercept_:.2f}")
    print(f"Predicted score for 7 hours: {new_prediction[0]:.1f}")
    print(f"R² Score: {r2_score(y, predictions):.3f}")
    print()

def house_price_example():
    """House price prediction example"""
    print("=== House Price Example ===")
    
    # Data: Size (sq ft) vs Price ($)
    sizes = np.array([[1000], [1200], [1500], [1800], [2000], [2200], [2500]])
    prices = np.array([200000, 240000, 300000, 360000, 400000, 440000, 500000])
    
    # Model
    model = LinearRegression()
    model.fit(sizes, prices)
    
    # Prediction for 1800 sq ft
    prediction = model.predict([[1800]])
    
    print(f"Price per sq ft: ${model.coef_[0]:.2f}")
    print(f"Base price: ${model.intercept_:.0f}")
    print(f"1800 sq ft house price: ${prediction[0]:,.0f}")
    print()

def salary_prediction():
    """Salary vs experience example"""
    print("=== Salary Prediction Example ===")
    
    # Data
    experience = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    salary = np.array([30000, 35000, 42000, 48000, 55000, 62000, 68000, 75000, 82000, 90000])
    
    # Model
    model = LinearRegression()
    model.fit(experience, salary)
    
    # Predictions
    predictions = model.predict(experience)
    
    print(f"Salary increase per year: ${model.coef_[0]:,.0f}")
    print(f"Starting salary: ${model.intercept_:,.0f}")
    print(f"R² Score: {r2_score(salary, predictions):.3f}")
    
    # Predict for 12 years experience
    future_salary = model.predict([[12]])
    print(f"Predicted salary for 12 years experience: ${future_salary[0]:,.0f}")
    print()

def visualize_regression():
    """Create a simple visualization"""
    print("=== Creating Visualization ===")
    
    # Generate sample data
    np.random.seed(42)
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    y = 2 * X.ravel() + 1 + np.random.normal(0, 2, 50)
    
    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.6, label='Data points')
    plt.plot(X, predictions, color='red', linewidth=2, label='Regression line')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title('Linear Regression Example')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig('/home/paradontex/Desktop/ML lectures/linear_regression_plot.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'linear_regression_plot.png'")
    plt.show()

if __name__ == "__main__":
    print("🚀 Linear Regression Demo - 3 Minute ML Lecture\n")
    
    basic_example()
    house_price_example()
    salary_prediction()
    visualize_regression()
    
    print("✅ Demo completed! Check the HTML lecture for detailed explanation.")
