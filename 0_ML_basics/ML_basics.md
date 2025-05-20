
---

**1. How to deal with overfitting?**
Use more data, simpler models, regularization (L1/L2), dropout, early stopping, pruning, or data augmentation.

**2. Describe the difference between bias and variance**
Bias: Error from overly simplistic models.
Variance: Error from overly complex models sensitive to noise.

**3. What is the role of regularization? Common regularization methods include L1 and L2**
Regularization penalizes model complexity to prevent overfitting.

* L1 (Lasso): promotes sparsity. $\lambda \sum^{N}_{i=1} | \theta_i |$
* L2 (Ridge): penalizes large weights smoothly. $\lambda \sum^{N}_{i=1} \theta_i^2 $

**4. Explain the two evaluation metrics PR-AUC and ROC-AUC**

* PR-AUC: Precision vs Recall, useful for imbalanced data.
* ROC-AUC: True Positive Rate vs False Positive Rate, measures overall classification performance.

**5. How to deal with imbalanced data?**
Use resampling (oversampling minority, undersampling majority), SMOTE, class weights, or specialized metrics (e.g., PR-AUC).

**6. What is hyperparameter optimization?**
Tuning non-learnable parameters (like learning rate or depth) to improve model performance using methods like grid search or Bayesian optimization.

**7. What are the common data split methods in machine learning?**

* Train/Validation/Test
* K-Fold Cross-Validation
* Stratified sampling (for class balance)

**8. Explain the principle of logistic regression**
A linear model using the sigmoid function to output probabilities for binary classification.

**9. Describe the principle of decision tree and its time complexity during inference**
Splits data by feature thresholds to minimize impurity (e.g., Gini).
Inference time: O(depth of the tree), typically O(log n) for balanced trees.

**10. What is the difference between Random Forest and XGBoost?**

* Random Forest: Bagging of decision trees, parallel, less prone to overfitting.
* XGBoost: Boosting with gradient descent, sequential, often more accurate.

**11. Compare the similarities and differences between bagging and boosting**

* Similarity: Both are ensemble methods.
* Bagging: Parallel, reduces variance.
* Boosting: Sequential, reduces bias.

**12. Describe the principle and shortcomings of K-means algorithm**
Clusters data by minimizing intra-cluster distance.
Shortcomings: Requires k, sensitive to initialization and outliers.

**13. How to implement KNN algorithm?**
Store all data points, compute distance to query point, vote among k nearest neighbors.

**14. Explain the principle and application scenarios of PCA**
Reduces dimensionality by projecting data onto top eigenvectors of the covariance matrix.
Used in noise reduction, visualization, and preprocessing.

**15. Define SVM and describe its optimization process**
SVM finds the optimal hyperplane maximizing margin between classes.
Optimization uses quadratic programming with constraints (hard or soft margin).

**16. What is p-value**
The p-value is the probability of observing a result at least as extreme as the one obtained, assuming the null hypothesis is true.
A small p-value (typically ≤ 0.05) suggests evidence against the null hypothesis.
e.g. A and B,
H0 (null): There is no linear relationship (slope = 0).
p-value for slope = 0.003 < 0.05 ⇒ reject null ⇒ significant linear relationship.

**17. Type 1 vs. Type 2 error**
Type I Error: Type I error occurs when the null hypothesis is true and we reject it. \
Type II Error: Type II error occurs when the null hypothesis is false and we accept it.

**18. Correlation and Covariance**
Correlation: Correlation tells us how strongly two random variables are related to each other. It takes values between -1 to +1. 
$$
Correlation = \frac{Cov(x,y)}{6x 6y}
$$
Covariance: Covariance tells us the direction of the linear relationship between two random variables. It can take any value between -∞ and +∞.
$$
Cov(x, y) = \frac{\sum (x_i - \overline{x})(y_i - \overline{y})}{N}
$$

**19. SVM**
Support Vectors are data points that are nearest to the hyperplane. It influences the position and orientation of the hyperplane. Removing the support vectors will alter the position of the hyperplane. The support vectors help us build our support vector machine model.

**20. What is Ensemble learning?**
Ensemble learning is a combination of the results obtained from multiple machine learning models to increase the accuracy for improved decision-making. 

**21. What is cross validation**
Cross-Validation in Machine Learning is a statistical resampling technique that uses different parts of the dataset to train and test a machine learning algorithm on different iterations. The aim of cross-validation is to test the model’s ability to predict a new set of data that was not used to train the model. Cross-validation avoids the overfitting of data.

**22. different methods to split a tree in a decision tree algorithm?**
- Variance: Splitting the nodes of a decision tree using the variance is done when the target variable is continuous.

- Information Gain: Splitting the nodes of a decision tree using Information Gain is preferred when the target variable is categorical. $IG = 1 - \text{Entropy}$, where $\text{Entropy} = - \sum p_i log_2 p_i$

- Gini Impurity: Splitting the nodes of a decision tree using Gini Impurity is followed when the target variable is categorical. $I_G (n) = 1 - \sum^n_{i=1} (p_i)^2$

**23. Primarily 5 assumptions for a Linear Regression model**
- Multivariate normality
- No auto-correlation
- Homoscedasticity
- Linear relationship
- No or little multicollinearity

**24. What is the difference between Lasso and Ridge regression?**
Lasso(also known as L1) and Ridge(also known as L2) regression are two popular regularization techniques that are used to avoid overfitting of data. These methods are used to penalize the coefficients to find the optimum solution and reduce complexity. The Lasso regression works by penalizing the sum of the absolute values of the coefficients. In Ridge or L2 regression, the penalty function is determined by the sum of the squares of the coefficients.

**25. Bagging and boosting**
Bagging is an ensemble technique that trains multiple models independently on random subsets of the training data (with replacement), and then averages (for regression) or votes (for classification) their predictions. (Reduce variance; Models are trained in parallel; Final output is majority vote or average)

Boosting is an ensemble technique where models are trained sequentially, and each new model tries to correct the errors made by the previous ones. (Reduce bias; Models are trained in sequence; Each model is weighted in the final prediction; Focus on hard-to-predict examples)

