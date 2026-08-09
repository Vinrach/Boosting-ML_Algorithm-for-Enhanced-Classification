# 🧠 AdaBoost Ensemble Learning for Breast Cancer Classification

### From-Scratch Implementation of Adaptive Boosting with Decision Stumps for Binary Medical Classification

<p align="center">

<img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/AdaBoost-Ensemble%20Learning-FF6F00?style=for-the-badge" alt="AdaBoost">
<img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
<img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
<img src="https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib">
<img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter">

</p>

---

## 📌 Overview

This project implements the **AdaBoost (Adaptive Boosting)** ensemble learning
algorithm from scratch in Python using **decision stumps as weak learners**.

The model is applied to the **Breast Cancer Wisconsin Diagnostic Dataset**
to perform binary classification of tumors as:

* **Malignant (M)**
* **Benign (B)**

Instead of relying on a pre-built AdaBoost classifier, the implementation
manually constructs the boosting process, including:

* Decision-stump generation
* Weighted error calculation
* Weak-classifier weighting
* Sample-weight updates
* Ensemble prediction
* Hyperparameter evaluation
* Feature-importance analysis

The project demonstrates how multiple weak classifiers can be combined
sequentially to create a stronger classification model.

---

# 🎯 Problem Statement

Breast cancer diagnosis involves distinguishing between malignant and benign
tumors based on measurable characteristics of cell nuclei.

The objective of this project is to build a binary classification model that
predicts whether a tumor is:

```text
M → Malignant
B → Benign
```

using numerical features from the Breast Cancer Wisconsin Diagnostic Dataset.

The project investigates whether an **AdaBoost ensemble of simple decision
stumps** can learn useful decision boundaries and accurately classify tumor
observations.

---

# 🧠 Why AdaBoost?

AdaBoost is an ensemble learning technique that combines multiple weak learners
into a stronger classifier.

The key idea is to train weak learners sequentially while increasing the
importance of observations that were incorrectly classified by previous
learners.

```text
                 Training Dataset
                        │
                        ▼
              Initialize Equal Weights
                        │
                        ▼
                Train Decision Stump
                        │
                        ▼
                Calculate Weighted Error
                        │
                        ▼
             Calculate Classifier Weight
                        │
                        ▼
            Increase Misclassified Weights
                        │
                        ▼
                 Normalize Weights
                        │
                        ▼
                Train Next Stump
                        │
                       ...
                        │
                        ▼
              Weighted Ensemble Model
                        │
                        ▼
                   Final Prediction
```

This allows subsequent weak learners to focus more heavily on difficult or
previously misclassified observations.

---

# 🏗️ Model Architecture

The implementation uses **decision stumps** as the base learners.

A decision stump is a one-level decision tree that makes a classification
decision using:

* One feature
* One threshold
* One inequality direction

```text
                  Feature
                     │
                Threshold
                     │
              ┌──────┴──────┐
              │             │
           Condition      Condition
              │             │
              ▼             ▼
          Class -1       Class +1
```

Multiple stumps are then combined using their calculated AdaBoost weights.

---

# 🔬 Implementation Workflow

## 1. Dataset Loading

The Breast Cancer Wisconsin dataset is loaded using Pandas:

```python
data = pd.read_csv("breast-cancer-wisconsin-data.csv")
```

The dataset is provided directly in the repository.

---

## 2. Data Preprocessing

The `id` column is removed because it does not represent a predictive feature.

The remaining numerical columns are used as model inputs.

The target variable is converted into binary numerical labels:

```text
Malignant →  1
Benign    → -1
```

This representation is suitable for the AdaBoost prediction formulation used
in the implementation.

---

# 📊 Dataset Features

The model uses numerical diagnostic measurements describing characteristics
of cell nuclei.

Examples include:

* Radius
* Texture
* Perimeter
* Area
* Smoothness
* Compactness
* Concavity
* Concave points
* Symmetry
* Fractal dimension

These measurements provide the numerical inputs used by the decision stumps.

---

# 🔀 Train-Test Split

The dataset is manually divided into:

```text
70% → Training
30% → Testing
```

A fixed NumPy random seed is used:

```python
np.random.seed(42)
```

This makes the train/test partition reproducible.

```text
                 Full Dataset
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
       70% Training          30% Testing
             │                   │
             ▼                   ▼
        AdaBoost Model       Evaluation
```

---

# 🌳 Decision Stump Implementation

The decision stump is implemented from scratch.

For every feature, the algorithm:

1. Identifies unique feature values.
2. Calculates candidate thresholds using midpoints.
3. Tests both inequality directions.
4. Calculates weighted classification error.
5. Selects the split with the lowest weighted error.

The implementation therefore searches across feature dimensions and thresholds
rather than relying on a pre-built tree classifier.

Conceptually:

```text
Feature 1
 ├── Threshold A
 ├── Threshold B
 └── Threshold C

Feature 2
 ├── Threshold A
 ├── Threshold B
 └── Threshold C

...

Feature N
 ├── Threshold A
 ├── Threshold B
 └── Threshold C

          │
          ▼

Lowest Weighted Error
          │
          ▼

Best Decision Stump
```

---

# ⚡ AdaBoost Implementation

The AdaBoost implementation initializes all training observations with equal
weights:

```python
w = np.ones(n) / n
```

For every boosting iteration, the algorithm:

1. Trains the best decision stump under the current sample weights.
2. Calculates the stump's weighted classification error.
3. Calculates the weak classifier's `alpha` weight.
4. Stores the weak classifier.
5. Updates sample weights.
6. Normalizes the new sample weights.
7. Repeats for the configured number of estimators.

The classifier weight is calculated as:

```python
classifier_weight = 0.5 * np.log(
    (1 - error) / (error + eps)
)
```

The sample weights are then updated according to the classifier's performance
and normalized before the next iteration.

---

# 🔁 Adaptive Sample Weighting

The core mechanism of AdaBoost is adaptive sample weighting.

Initially:

```text
All observations
      │
      ▼
Equal weights
```

After each weak learner:

```text
Correctly classified
        │
        ▼
Lower relative importance

Incorrectly classified
        │
        ▼
Higher relative importance
```

This encourages subsequent decision stumps to focus on observations that were
difficult for earlier classifiers.

---

# 🧮 Ensemble Prediction

Each decision stump generates a prediction.

The final prediction is calculated as the weighted combination of all weak
classifiers:

```text
Final Score =
    α₁ × Stump₁
  + α₂ × Stump₂
  + α₃ × Stump₃
  + ...
  + αₙ × Stumpₙ
```

The final class is determined from the sign of the combined prediction.

The implementation explicitly aggregates each stump's prediction using its
corresponding `alpha` weight.

---

# 🔧 Hyperparameter Tuning

The project evaluates multiple combinations of:

* `n_estimators`
* `learning_rate`

The tested configurations are:

| Number of Estimators | Learning Rate |
| -------------------: | ------------: |
|                   50 |           1.0 |
|                  100 |           0.5 |
|                  200 |           0.1 |
|                   50 |           0.1 |

Each configuration is trained on the training set and evaluated on the
held-out test set.

The configuration with the highest test accuracy is selected.

---

# 🏆 Results

The best-performing configuration in the recorded notebook execution was:

| Hyperparameter  | Best Value |
| --------------- | ---------: |
| `n_estimators`  |    **100** |
| `learning_rate` |    **0.5** |

The resulting test accuracy was:

```text
97.66%
```

The final model trained using the selected hyperparameters also achieved:

```text
Final Accuracy: 97.66%
```

These values are the results recorded in the committed notebook execution.

---

# 📈 Feature Importance

The project also analyzes feature importance using the cumulative weights
(`alpha`) assigned to the weak classifiers.

This provides an indication of which features were used most heavily across
the ensemble.

The analysis identified **`texture_mean`** among the higher-importance
features in the recorded experiment.

Conceptually:

```text
Decision Stumps
      │
      ▼
Feature Usage
      │
      ▼
Classifier Weights (α)
      │
      ▼
Cumulative Feature Importance
      │
      ▼
Feature Importance Visualization
```

This adds an interpretability component to the classification analysis.

---

# 🔎 Key Findings

The experiment demonstrates that:

* A collection of simple decision stumps can form a strong ensemble classifier.
* Adaptive sample weighting focuses subsequent learners on difficult examples.
* Hyperparameter selection affects the resulting classification performance.
* The implemented AdaBoost model achieved **97.66% test accuracy** in the
  recorded experiment.
* Feature-weight analysis can provide additional insight into which variables
  contribute to the ensemble's decisions.

---

# 🧪 Machine Learning Workflow

```text
Breast Cancer Dataset
          │
          ▼
     Data Cleaning
          │
          ▼
   Feature / Target Split
          │
          ▼
     Binary Encoding
     M → +1
     B → -1
          │
          ▼
     Train/Test Split
       70% / 30%
          │
          ▼
 Initialize Sample Weights
          │
          ▼
   Decision Stump Search
          │
          ▼
  Weighted Error Calculation
          │
          ▼
     Alpha Calculation
          │
          ▼
    Weight Recalculation
          │
          ▼
    Repeat Boosting
          │
          ▼
   Weighted Predictions
          │
          ▼
       Evaluation
          │
          ▼
 Feature Importance Analysis
```

---

# 🛠️ Technology Stack

| Technology                          | Purpose                                       |
| ----------------------------------- | --------------------------------------------- |
| **Python**                          | Model implementation                          |
| **NumPy**                           | Numerical computation and model operations    |
| **Pandas**                          | Dataset loading and preprocessing             |
| **Matplotlib**                      | Visualization and feature-importance analysis |
| **Jupyter Notebook**                | Interactive development and experimentation   |
| **AdaBoost**                        | Ensemble learning algorithm                   |
| **Decision Stumps**                 | Weak/base classifiers                         |
| **Breast Cancer Wisconsin Dataset** | Binary classification dataset                 |

The actual notebook imports Pandas, NumPy, Random, Itertools, and Matplotlib.

---

# 📂 Repository Structure

```text
AdaBoost/
│
├── Adaboost.ipynb
├── breast-cancer-wisconsin-data.csv
└── README.md
```

### `Adaboost.ipynb`

Contains the complete implementation:

* Dataset loading
* Data preprocessing
* Train/test splitting
* Decision stump implementation
* AdaBoost implementation
* Prediction
* Hyperparameter evaluation
* Final model training
* Accuracy evaluation
* Feature-importance analysis

### `breast-cancer-wisconsin-data.csv`

Breast Cancer Wisconsin Diagnostic Dataset used for binary tumor
classification.

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/Vinrach/AdaBoost.git

cd AdaBoost
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install numpy pandas matplotlib jupyter
```

## 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Adaboost.ipynb
```

and execute the notebook cells sequentially.

---

# 🔁 Reproducibility

The train/test split uses:

```python
np.random.seed(42)
```

This ensures that the dataset partition is reproducible across executions.

For a more production-oriented implementation, additional deterministic
controls could be introduced around any other randomized operations.

---

# ⚠️ Limitations

This project is an educational implementation intended to demonstrate the
AdaBoost algorithm and ensemble-learning concepts.

Important limitations include:

* Evaluation is based primarily on accuracy.
* Only a small set of hyperparameter combinations is evaluated.
* No cross-validation is currently implemented.
* No confusion matrix is reported.
* Precision, recall, F1-score, and ROC-AUC are not currently evaluated.
* The implementation uses a manual train/test split.
* The decision-stump search can become computationally expensive as dataset
  size and feature dimensionality increase.
* The feature-importance analysis is based on cumulative classifier weights
  rather than a formal model-agnostic explainability method.

These are natural areas for future development.

---

# 🔮 Future Improvements

## Model Evaluation

* Add stratified cross-validation.
* Report precision, recall, F1-score, and ROC-AUC.
* Add confusion matrix visualization.
* Plot ROC and Precision-Recall curves.
* Compare training and validation performance.

## Model Development

* Compare from-scratch AdaBoost against `sklearn.AdaBoostClassifier`.
* Experiment with different weak learners.
* Expand the hyperparameter search.
* Implement early stopping.
* Investigate class imbalance handling.

## Explainability

* Improve feature-importance analysis.
* Compare AdaBoost feature weights with permutation importance.
* Add SHAP-based model interpretation.
* Analyze false-positive and false-negative cases.

## Engineering

* Extract the AdaBoost implementation into reusable Python modules.
* Add unit tests.
* Add a `requirements.txt`.
* Add experiment configuration.
* Add automated model evaluation.
* Add CI testing through GitHub Actions.

---

# 🎓 Academic Context

This project was developed as a machine learning implementation exercise focused
on **ensemble learning and AdaBoost classification**.

The implementation emphasizes understanding the internal mechanics of boosting
rather than treating AdaBoost as a black-box library algorithm.

The project therefore demonstrates both:

```text
Machine Learning Theory
        +
Algorithm Implementation
        +
Experimental Evaluation
```

---

# 🧠 Skills Demonstrated

### Machine Learning

* Supervised learning
* Binary classification
* Ensemble learning
* Boosting
* AdaBoost
* Decision stumps
* Hyperparameter tuning

### Data Science

* Data preprocessing
* Train/test splitting
* Model evaluation
* Feature analysis
* Data visualization
* Reproducible experimentation

### Python

* NumPy
* Pandas
* Matplotlib
* Jupyter Notebook
* Algorithm implementation from scratch

---

# 📚 Reference

The implementation follows the standard AdaBoost principle of combining
weighted weak learners into a stronger ensemble classifier.

The project focuses specifically on implementing the algorithm mechanics using
decision stumps rather than relying on a pre-built AdaBoost implementation.

---

# 👨‍💻 Author

**Vinrach**

Master's Degree Project

GitHub: [@Vinrach](https://github.com/Vinrach)

---

<div align="center">

### Machine Learning • Ensemble Learning • AdaBoost • Classification

**Understanding Boosting by Implementing It From Scratch**

</div>
