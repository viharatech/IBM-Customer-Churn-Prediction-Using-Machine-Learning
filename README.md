# 📊 IBM Customer Churn Prediction Using Machine Learning

## 📘 Project Overview

This project aims to predict **customer churn** for a telecom company using the **IBM Sample Dataset**.
The objective is to identify which customers are likely to discontinue services, enabling proactive retention strategies.

We conducted **Exploratory Data Analysis (EDA)**, **Feature Engineering**, **Data Balancing**, **Model Training**, and **Web Deployment** using modern data science techniques.

---

## 📂 Dataset Overview

* **Source:** IBM Sample Telco Customer Churn Dataset
* **Rows:** 7043
* **Columns:** 21 features + Target (`Churn`)
* **Target Variable:** `Churn` (Yes = Customer left, No = Customer stayed)

### Key Features

| Feature                              | Description                          |
| ------------------------------------ | ------------------------------------ |
| Gender                               | Male / Female                        |
| Partner / Dependents                 | Customer relationship info           |
| InternetService                      | DSL / Fiber optic / None             |
| PaymentMethod                        | Electronic check, Credit card, etc.  |
| Contract                             | Month-to-month / One year / Two year |
| Tenure, MonthlyCharges, TotalCharges | Numeric variables                    |
| Churn                                | Target variable (Yes/No)             |

---

## ⚙️ Workflow Summary

### 1️⃣ Exploratory Data Analysis (EDA)

* Visualized **demographic trends** and **service usage** patterns.
* Observed that:

  * Customers without dependents and partners churned more.
  * Fiber optic customers had higher churn rates.
  * Month-to-month contracts had maximum churn.

### 2️⃣ Handling Missing Values

Used multiple techniques and finalized **IterativeImputer** for robust imputation:

* `KNNImputer`
* `SimpleImputer`
* `IterativeImputer`
* `MeanMedianImputer`
* `ArbitraryNumberImputer`
* `RandomSampleImputer`

**Finalized:** `IterativeImputer` (DecisionTreeRegressor model) — due to better statistical consistency and performance.

---

### 3️⃣ Variable Transformation

Performed data transformation to improve model interpretability and reduce skewness.
Tested:

* Log, SquareRoot, CubeRoot, Power, Quantile, and Tanh transforms.

**Finalized:** `QuantileTransformer` (Normal distribution)
→ Produced bell-curve-like distribution and reduced skewness effectively.

---

### 4️⃣ Outlier Handling

Techniques compared:

* **Winsorizer (IQR, Gaussian, Quantile, MAD)**
* **Isolation Forest**
* **Quantile Clipping (0.01–0.99)**

**Finalized:** `Gaussian Winsorizer` → smooth, symmetric capping aligned with scaled numeric data.

---

### 5️⃣ Encoding Categorical Features

Applied a hybrid approach:

| Type                 | Method                               | Example Columns                         |
| -------------------- | ------------------------------------ | --------------------------------------- |
| **Ordinal Encoding** | Contract                             | Month-to-month < One year < Two year    |
| **One-Hot Encoding** | InternetService, PaymentMethod, etc. | Categorical variables                   |
| **Manual Mapping**   | sim                                  | {'Jio':0, 'Airtel':1, 'Vi':2, 'BSNL':3} |
| **Label Encoding**   | Target (Churn)                       | Yes → 1, No → 0                         |

---

### 6️⃣ Feature Selection

Applied **Filter Methods**:

* VarianceThreshold
* Chi-Square Test
* ANOVA F-Test
* t-Test
* SelectKBest

These helped retain only statistically significant features with the highest predictive power.

---

### 7️⃣ Correlation & Hypothesis Testing

* Used Pearson correlation to remove multicollinearity (r > 0.85).
* Conducted **Chi², t-Test, ANOVA** to validate feature significance.

---

### 8️⃣ Balancing the Dataset

Used **SMOTE (Synthetic Minority Oversampling Technique)** from `imblearn`:

* Balanced Churn vs Non-Churn ratio from **27% : 73%** to **50% : 50%**
* Resulted in better recall and F1-score for churn detection.

---

### 9️⃣ Feature Scaling

Used **StandardScaler** for standardization:

[
z = \frac{x - \mu}{\sigma}
]

* Mean = 0, Standard Deviation = 1
* Ensures equal contribution of all numeric features.
* Works best with algorithms like Logistic Regression & SVM.

---

### 🔟 Model Training & Evaluation

#### Models Tested

| Model               | Type           | Key Trait                       |
| ------------------- | -------------- | ------------------------------- |
| Logistic Regression | Linear         | Fast, interpretable             |
| KNN Classifier      | Distance-based | Simple but sensitive to scaling |
| GaussianNB          | Probabilistic  | Handles high-dimensional data   |
| Decision Tree       | Tree-based     | Easy to interpret               |
| Random Forest       | Ensemble       | Robust, less overfitting        |
| XGBoost             | Boosting       | High accuracy                   |
| SVC                 | Margin-based   | Works well for non-linear data  |
| Gradient Boosting   | Boosting       | High predictive power           |

#### Metrics Used

* Accuracy
* Precision
* Recall
* F1-Score
* ROC–AUC Score

#### Example Results

| Model               | Accuracy | ROC-AUC |
| ------------------- | -------- | ------- |
| Logistic Regression | 0.75     | 0.86    |
| Random Forest       | 0.71     | 0.66    |
| XGBoost             | 0.69     | 0.67    |
| Gradient Boosting   | 0.79     | 0.86    |

**Finalized Model:** Gradient Boosting Classifier

> Balanced accuracy, interpretability, and computational efficiency.

---

## 📈 ROC and AUC

* ROC curve demonstrates model’s ability to distinguish between churn and non-churn.
* AUC close to **0.86–0.88** indicates excellent discriminative capability.

---

## 🌐 Model Deployment

### Platform

* **Backend:** Flask
* **Hosting:** Render Cloud Platform

### Deployment Workflow

1. Trained model & scaler saved as `model.pkl` and `scaler.pkl`.
2. Flask app built to handle user inputs and predictions.
3. Auto-deployed via GitHub → Render integration.
4. Live churn prediction available through web interface.

---

## 🧰 Tools & Libraries

* **Python 3.9+**
* **NumPy, Pandas, Matplotlib, Seaborn**
* **scikit-learn**
* **feature-engine**
* **imblearn**
* **xgboost**
* **Flask**
* **Render (for deployment)**

---

## 📊 Key Insights

* Customers with **month-to-month contracts** churn more frequently.
* **Fiber optic** users have higher churn rates.
* Customers **without dependents or partners** are more likely to leave.
* **Electronic check** payment method correlates with higher churn.
* **Tenure** and **contract length** are strong churn predictors.

---

## 🚀 Results Summary

* AUC ≈ **0.88**
* Accuracy ≈ **79%**
* Balanced recall and precision after SMOTE
* Smooth, explainable prediction pipeline

---

## 👨‍💻 Author

**Madhala. Siva Narayana Surya Chandra**
Machine Learning Enthusiast | Data Science Practitioner

For support : sivanarayana9347@gmail.com 
Find the Project here : https://ibm-customer-churn-prediction-using.onrender.com


