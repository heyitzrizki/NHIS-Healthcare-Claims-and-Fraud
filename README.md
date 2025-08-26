# Healthcare Fraud Detection Project

Healthcare Fraud Detection using Machine Learning  
by **Rizki Anwar Syaifullah**

**Dataset Source:** [Kaggle – NHIS Healthcare Claims and Fraud Dataset](https://www.kaggle.com/datasets/bonifacechosen/nhis-healthcare-claims-and-fraud-dataset)

---

## 📌 Problem Statement
Healthcare fraud is a major challenge for insurers and healthcare providers, leading to **financial losses, distorted resource allocation, and reduced trust** in the healthcare system. Identifying fraudulent claims early is crucial to minimize risks and improve audit efficiency.

---

## 🎯 Goals
Our project aims to:
1. **Detect fraudulent claims** → Build ML models that can classify legitimate vs fraudulent claims.  
2. **Handle imbalanced data** → Apply resampling techniques like **SMOTE** to improve model performance on minority class (fraud).  
3. **Prioritize Recall** → Ensure most fraud cases are detected, even at the cost of higher false positives.  

---

## 📊 Metric Evaluation
- **Precision** → Accuracy of fraud predictions.  
- **Recall (Sensitivity)** → *Primary focus*, ensure fraudulent claims are not missed.  
- **F1 Score** → Balance between precision and recall.  
- **ROC-AUC** → Model discrimination ability.  
- **Accuracy** → General performance, but not the main focus due to imbalance.  

---

## ⚙️ Prerequisites
Make sure you have the following installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost imbalanced-learn joblib
