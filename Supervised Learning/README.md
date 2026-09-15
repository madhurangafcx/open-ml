# Supervised Learning

In **Supervised Learning**, algorithms learn a mapping function from input features ($X$) to target labels ($y$) using a labeled dataset. Once trained, the model can predict targets for new, unseen data.

$$\hat{y} = f(X)$$

---

## Core Problem Types

### 1. Regression
Predicting continuous numerical outcomes (e.g., house prices, temperature, stock trends).
* **Linear Regression**: Models a linear relationship between independent input variables and a continuous dependent variable.
* **Ridge / Lasso Regression**: Regularized linear regression methods that penalize large coefficients to avoid overfitting.

### 2. Classification
Categorizing inputs into discrete classes or labels (e.g., spam vs. ham, tumor benign vs. malignant).
* **Logistic Regression**: Estimates class probabilities using the logistic sigmoid function (binary and multinomial).
* **Support Vector Machines (SVM)**: Finds the optimal separating hyperplane that maximizes the margin between classes, utilizing kernel functions for non-linear boundaries.
* **Decision Trees & Random Forests**: Rule-based hierarchical trees and ensemble bagged trees that handle non-linear relationships with high interpretability and resilience to overfitting.
* **K-Nearest Neighbors (KNN)**: Non-parametric, instance-based method that assigns labels based on majority vote among closest neighbors.
* **Naive Bayes**: Probabilistic classifier based on Bayes' Theorem under the conditional independence assumption.

---

## Directory Structure & Available Notebooks

| Topic | Folder | Notebook / Lab | Status |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | [`Linear_regression/`](./Linear_regression/) | • [`Linear_regression.ipynb`](./Linear_regression/Linear_regression.ipynb)<br>• [`Programming Exercise 1 - Linear Regression_Full.ipynb`](./Linear_regression/Programming%20Exercise%201%20-%20Linear%20Regression_Full.ipynb) | :white_check_mark: Completed |
| **Support Vector Machines** | [`Support Vector Machine/`](./Support%20Vector%20Machine/) | • [`SVM Lab sheet - modify.ipynb`](./Support%20Vector%20Machine/SVM%20Lab%20sheet%20-%20modify.ipynb) | :white_check_mark: Completed |
| **Logistic Regression** | [`logistic_regression/`](./logistic_regression/) | *Notebook coming soon* | :hourglass: In Progress |
| **Decision Trees & Ensembles** | `Decision_Trees/` | *Planned* | :black_square_button: Planned |

---
