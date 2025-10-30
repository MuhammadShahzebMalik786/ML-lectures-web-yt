# Ensemble Methods - Quiz & Exercises

## 📝 Quick Knowledge Check (5 minutes)

### Multiple Choice Questions

**1. What is the main principle behind ensemble methods?**
a) Using the most complex model available
b) Combining multiple models to improve performance
c) Always using deep learning
d) Reducing the size of the dataset

**2. Which ensemble method trains models in parallel?**
a) Boosting
b) AdaBoost
c) Bagging
d) Sequential learning

**3. What does "bootstrap" mean in bootstrap aggregating?**
a) Starting the computer
b) Sampling with replacement
c) Using the best model
d) Removing outliers

**4. Which ensemble method is most likely to overfit?**
a) Random Forest
b) Bagging
c) Boosting
d) Simple averaging

**5. In stacking, what is the role of the meta-learner?**
a) To replace all base models
b) To learn how to combine base model predictions
c) To select the best base model
d) To preprocess the data

### True/False Questions

**6. T/F: Ensemble methods always perform better than individual models.**

**7. T/F: Random Forest uses both bootstrap sampling and feature randomness.**

**8. T/F: AdaBoost gives equal weight to all training samples throughout training.**

**9. T/F: Stacking requires cross-validation to avoid overfitting.**

**10. T/F: Boosting reduces bias while bagging reduces variance.**

---

## 🧠 Conceptual Understanding Questions

### Short Answer (2-3 sentences each)

**1. Explain why diversity is important in ensemble methods.**

**2. When would you choose boosting over bagging?**

**3. What are the computational trade-offs of using ensemble methods?**

**4. How does Random Forest prevent overfitting compared to a single decision tree?**

**5. Describe a real-world scenario where ensemble thinking is naturally used.**

---

## 💻 Practical Exercises

### Exercise 1: Bootstrap Sampling Analysis (15 minutes)
```python
# Complete this code to analyze bootstrap sampling
import numpy as np

def analyze_bootstrap_diversity(original_data, n_samples=100):
    """
    Analyze how bootstrap sampling creates diversity
    
    TODO:
    1. Create n_samples bootstrap samples
    2. Calculate how many unique values each sample contains
    3. Calculate the overlap between samples
    4. Return statistics about diversity
    """
    # Your code here
    pass

# Test with: analyze_bootstrap_diversity(np.arange(1, 21))
```

### Exercise 2: Manual Voting Classifier (20 minutes)
```python
# Implement a simple voting classifier from scratch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

class SimpleVotingClassifier:
    def __init__(self, models):
        self.models = models
        
    def fit(self, X, y):
        """
        TODO: Train all models on the data
        """
        # Your code here
        pass
        
    def predict(self, X):
        """
        TODO: Get predictions from all models and return majority vote
        """
        # Your code here
        pass

# Test your implementation
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

models = [
    DecisionTreeClassifier(),
    GaussianNB(),
    SVC()
]

voting_clf = SimpleVotingClassifier(models)
# Test your implementation here
```

### Exercise 3: Ensemble Performance Analysis (25 minutes)
```python
# Compare different ensemble methods on a dataset of your choice
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

def compare_ensemble_methods(X, y):
    """
    TODO:
    1. Split data into train/test
    2. Train individual models and ensemble methods
    3. Compare performance using cross-validation
    4. Create a performance comparison table
    5. Analyze which method works best and why
    """
    # Your code here
    pass

# Test with wine dataset
X, y = load_wine(return_X_y=True)
compare_ensemble_methods(X, y)
```

### Exercise 4: Stacking Implementation (30 minutes)
```python
# Implement a simple stacking classifier
from sklearn.model_selection import KFold
import numpy as np

class SimpleStackingClassifier:
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
        
    def fit(self, X, y, cv=5):
        """
        TODO:
        1. Use cross-validation to generate base model predictions
        2. Train base models on full dataset
        3. Train meta-model on base model predictions
        """
        # Your code here
        pass
        
    def predict(self, X):
        """
        TODO:
        1. Get predictions from all base models
        2. Use meta-model to make final prediction
        """
        # Your code here
        pass

# Test your stacking implementation
```

---

## 🎯 Advanced Challenges

### Challenge 1: Ensemble Diversity Analysis
Create a function that measures the diversity between models in an ensemble and shows how diversity relates to ensemble performance.

### Challenge 2: Dynamic Ensemble
Implement an ensemble that dynamically selects which models to use based on the input features.

### Challenge 3: Ensemble Interpretability
Create a method to explain ensemble predictions by showing the contribution of each base model.

---

## 📊 Mini-Project: Ensemble Method Comparison

### Project Description
Compare the performance of different ensemble methods on multiple datasets and analyze when each method works best.

### Requirements:
1. **Datasets**: Use at least 3 different datasets (classification)
2. **Methods**: Compare bagging, boosting, and stacking
3. **Metrics**: Accuracy, precision, recall, F1-score
4. **Analysis**: 
   - Which method works best on which type of data?
   - How does ensemble size affect performance?
   - What are the computational costs?

### Deliverables:
1. Python script with complete implementation
2. Results table comparing all methods
3. Visualization of performance differences
4. Written analysis (2-3 paragraphs) explaining findings

---

## ✅ Answer Key

### Multiple Choice Answers:
1. b) Combining multiple models to improve performance
2. c) Bagging
3. b) Sampling with replacement
4. c) Boosting
5. b) To learn how to combine base model predictions

### True/False Answers:
6. **False** - Ensembles don't always perform better, especially if base models are highly correlated
7. **True** - Random Forest uses both bootstrap sampling and random feature selection
8. **False** - AdaBoost adjusts sample weights based on previous errors
9. **True** - Cross-validation prevents the meta-learner from overfitting to base model predictions
10. **True** - This is a key distinction between the two approaches

### Short Answer Sample Responses:

**1. Diversity importance:**
Diversity ensures that models make different types of errors, allowing the ensemble to compensate for individual model weaknesses. If all models make the same mistakes, the ensemble provides no benefit over a single model.

**2. Boosting vs Bagging:**
Choose boosting when you have high bias (underfitting) problems, as boosting focuses on reducing bias by learning from mistakes. Choose bagging when you have high variance (overfitting) problems.

**3. Computational trade-offs:**
Ensemble methods require training and storing multiple models, increasing computational cost and memory usage. However, bagging can be parallelized, while boosting must be sequential. The trade-off is often worth it for improved accuracy.

**4. Random Forest overfitting prevention:**
Random Forest prevents overfitting through bootstrap sampling (reducing correlation between trees) and random feature selection (increasing diversity). The averaging of many trees smooths out individual tree overfitting.

**5. Real-world ensemble example:**
Medical diagnosis often uses ensemble thinking - multiple doctors provide opinions, diagnostic tests give different information, and the final diagnosis combines all evidence. This reduces the chance of misdiagnosis from any single source.
