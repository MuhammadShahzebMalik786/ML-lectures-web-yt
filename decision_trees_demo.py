#!/usr/bin/env python3
"""
Decision Trees Demo - 3 Minute ML Lecture
Simple examples to understand decision trees
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def loan_approval_example():
    """Simple loan approval decision tree"""
    print("=== Loan Approval Example ===")
    
    # Features: [Age, Income, Credit_Score]
    X = np.array([
        [25, 50000, 650], [35, 80000, 720], [22, 30000, 580],
        [45, 90000, 750], [28, 45000, 620], [38, 70000, 680],
        [32, 60000, 700], [29, 40000, 590]
    ])
    # Labels: 0=Reject, 1=Approve
    y = np.array([0, 1, 0, 1, 0, 1, 1, 0])
    
    # Train model
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X, y)
    
    # Test predictions
    test_cases = [
        [30, 55000, 680],  # Should approve
        [20, 25000, 550],  # Should reject
        [40, 75000, 720]   # Should approve
    ]
    
    for i, case in enumerate(test_cases):
        prediction = tree.predict([case])
        result = "APPROVED" if prediction[0] else "REJECTED"
        print(f"Applicant {i+1}: Age={case[0]}, Income=${case[1]}, Credit={case[2]} → {result}")
    
    print()

def email_spam_detection():
    """Email spam detection using decision trees"""
    print("=== Email Spam Detection ===")
    
    # Features: [word_count, num_links, exclamation_marks, has_urgent_words]
    emails = np.array([
        [50, 1, 0, 0],    # Normal
        [200, 8, 15, 1],  # Spam
        [30, 0, 1, 0],    # Normal
        [180, 5, 12, 1],  # Spam
        [80, 2, 2, 0],    # Normal
        [250, 10, 20, 1], # Spam
        [40, 1, 0, 0],    # Normal
        [300, 15, 25, 1]  # Spam
    ])
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])  # 0=Normal, 1=Spam
    
    # Train spam detector
    spam_tree = DecisionTreeClassifier(max_depth=4, random_state=42)
    spam_tree.fit(emails, labels)
    
    # Test new emails
    new_emails = [
        [60, 1, 1, 0],    # Likely normal
        [150, 6, 10, 1],  # Likely spam
        [35, 0, 0, 0]     # Likely normal
    ]
    
    for i, email in enumerate(new_emails):
        prediction = spam_tree.predict([email])
        result = "SPAM" if prediction[0] else "NORMAL"
        print(f"Email {i+1}: Words={email[0]}, Links={email[1]}, Exclamations={email[2]} → {result}")
    
    # Show decision rules
    print("\nDecision Rules (first few):")
    rules = export_text(spam_tree, feature_names=['Words', 'Links', 'Exclamations', 'Urgent'])
    print(rules[:300] + "...")
    print()

def weather_prediction():
    """Weather-based activity prediction"""
    print("=== Weather Activity Prediction ===")
    
    # Features: [Temperature, Humidity, Wind_Speed]
    weather = np.array([
        [85, 85, 5],  [80, 90, 10], [83, 78, 3],  [70, 96, 2],
        [68, 80, 8],  [65, 70, 15], [64, 65, 12], [72, 95, 4],
        [69, 70, 6],  [75, 80, 7],  [75, 70, 11], [72, 90, 9]
    ])
    # Activity: 0=Stay Inside, 1=Go Outside
    activity = np.array([0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        weather, activity, test_size=0.3, random_state=42)
    
    # Train model
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Model Accuracy: {accuracy:.2f}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Test with new weather conditions
    new_weather = [[78, 75, 8]]  # Moderate conditions
    prediction = model.predict(new_weather)
    activity_name = "Go Outside" if prediction[0] else "Stay Inside"
    print(f"Weather: 78°F, 75% humidity, 8mph wind → {activity_name}")
    print()

def visualize_tree_structure():
    """Show how decision tree makes decisions"""
    print("=== Tree Structure Example ===")
    
    # Simple dataset
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [3, 3], [3, 4]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    # Train simple tree
    tree = DecisionTreeClassifier(max_depth=2, random_state=42)
    tree.fit(X, y)
    
    # Show tree rules
    rules = export_text(tree, feature_names=['Feature_1', 'Feature_2'])
    print("Complete Decision Tree Structure:")
    print(rules)
    
    # Test predictions
    test_points = [[1.5, 1.5], [2.5, 2.5], [3.5, 3.5]]
    for point in test_points:
        pred = tree.predict([point])
        print(f"Point {point} → Class {pred[0]}")

if __name__ == "__main__":
    print("🌳 Decision Trees Demo - 3 Minute ML Lecture\n")
    
    loan_approval_example()
    email_spam_detection()
    weather_prediction()
    visualize_tree_structure()
    
    print("✅ Demo completed! Decision trees are intuitive and powerful!")
