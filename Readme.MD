# 🏦 Bank Fraud Detection using Temporal Convolutional Networks (TCNs)

## 📌 Overview

This project explores the use of **Temporal Convolutional Networks (TCNs)** combined with RNN or Attention mechanisms to detect fraudulent bank transactions from time-series data. The goal is to better understand evolving fraud tactics by modeling transaction patterns over time, grouped by **months**.

> 💡 *"Detecting fraud isn't just about spotting anomalies—it's about understanding behavior over time."*

## 🔍 Motivation

Traditional machine learning models often fail to capture **temporal dynamics** in evolving datasets. We hypothesized that:
- Fraud patterns evolve monthly.
- TCNs can effectively model these temporal dependencies using **dilated causal convolutions**.
- Attention mechanisms or RNNs can further enhance temporal feature sensitivity.

## 📁 Dataset

- File used: `Base.csv`
- Number of rows: ~1,000,000
- Columns: 32
- Target variable: `fraud_bool` (1 = Fraud, 0 = Not Fraud)
- Key temporal feature: `month`

## 🔧 Data Preparation

### ✅ Cleaning
- Imputed missing **numerical** values using mean.
- Imputed **categorical** values using mode.

### 🔤 Encoding
- One-hot encoded all categorical columns.
- Standardized numerical features using `StandardScaler`.

### 📊 Feature Engineering
- Created **lag features** (1, 2, 3 months) for transaction behavior.
- Aggregated monthly features (mean, std, etc.)
- Created **rolling statistics** (mean/std of income, fraud rate, etc.)
- Introduced **ratios** like `income / intended_balcon_amount`.

## 🧠 Model Architectures

### 🎯 Attempt 1: TCN + RNN (LSTM)

#### Architecture:
- **TCN block** with dilated causal convolutions
- **MaxPooling1D**
- **LSTM layer** to capture sequential dependencies
- **Dense output** for regression (fraud probability)

> 🧪 *This model was prepared and structured for training but not fully executed due to setup limitations.*

### 🎯 Attempt 2: Random Forest Classifier

Used a traditional model as a baseline.

```python
from sklearn.ensemble import RandomForestClassifier
````

#### Results:

| Metric    | Score   |
| --------- | ------- |
| Accuracy  | \~0.989 |
| Precision | 0.0000  |
| Recall    | 0.0000  |
| F1 Score  | 0.0000  |
| AUC       | 0.5000  |

Despite a high **accuracy**, the model failed to identify any fraudulent transactions—**predicting all as non-fraud** due to extreme class imbalance.

## 📉 Evaluation

* Used standard classification metrics.
* Visualized the **confusion matrix**.
* Noted **model collapse** due to imbalanced data—no positive predictions made.

## 🚫 Limitations

* **Class imbalance**: Majority of the dataset consisted of non-fraud cases.
* RandomForestClassifier showed **zero recall**, missing all fraud cases.
* TCN-RNN architecture was defined but not executed end-to-end due to resource constraints.

## ✅ Key Takeaways

* Fraud detection needs **time-aware models** due to evolving behaviors.
* TCNs offer advantages over RNNs (parallelism, stable gradients).
* High accuracy does **not imply** good fraud detection performance.
* Class imbalance must be addressed via **oversampling**, **SMOTE**, or **cost-sensitive training**.

## 🚀 Future Work

* Fully train and evaluate the **TCN + RNN** or **TCN + Attention** hybrid.
* Implement **focal loss** or **class weights** to handle imbalance.
* Explore **sequence-based models** like Transformer Encoder.
* Deploy a real-time fraud monitoring dashboard.

## 🤝 Contributing

Want to contribute to improving fraud detection using AI/ML? Feel free to fork, suggest improvements, or reach out!
