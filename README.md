# Ice Cream Sales Prediction using Regression Pipelines

This repository implements an automated machine learning workflow to analyze and predict ice cream sales based on ambient temperature. It compares a baseline **Linear Regression** model against a **Polynomial Regression** model using scikit-learn pipelines.

---

## 📊 Dataset Overview
The model uses a temporal dataset containing temperature and sales figures recorded at specific times:
*   **Date / Time:** Timestamp of the observation.
*   **Temperature (Celsius):** Independent feature (X).
*   **Ice Cream Sales:** Target variable (y).

Data preprocessing confirms there are zero missing/null values across all features, ensuring clean training data.

---

## ⚙️ Project Architecture
The project utilizes `sklearn.pipeline.Pipeline` to streamline preprocessing and model execution without data leakage:

1.  **Data Splitting:** 80-20 train-test split (`random_state=42`).
2.  **Linear Pipeline:** `StandardScaler` ➡️ `LinearRegression`.
3.  **Polynomial Pipeline:** `PolynomialFeatures(degree=2)` ➡️ `StandardScaler` ➡️ `LinearRegression`.

---

## 📈 Model Performance Evaluation
After evaluating both pipelines on the unseen test set, the performance metrics are as follows:

| Regression Model | R² Score | RMSE | MAE |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | 0.8820 | 4.1157 | 3.6954 |
| **Polynomial Regression (Degree 2)** | **0.9273** | **3.2314** | **2.8061** |

### Key Insight:
The **Polynomial Regression** model performs significantly better. The higher R² score (0.9273) and lower error rates (RMSE/MAE) indicate that the relationship between temperature and ice cream sales is non-linear—sales accelerate or curve as temperatures rise.

---

## 🚀 How to Run & Predict

### Prerequisites
Make sure you have the required libraries installed:
```bash
pip install pandas numpy scikit-learn
```

### Inference Code Example
The notebook automatically selects the best performing model (`best_model`) to make new predictions:

```python
import pandas as pd

# Function to predict sales for any temperature
def predict_ice_cream_sales(temperature):
    new_data = pd.DataFrame([{"Temperature (Celsius)": temperature}])
    prediction = best_model.predict(new_data)[0]
    return prediction

# Example Prediction
temp = 30
predicted_sales = predict_ice_cream_sales(temperature=temp)
print(f"Predicted Ice Cream Sales at {temp}°C: {predicted_sales:.2f}")
# Output: Predicted Ice Cream Sales at 30°C: 28.22
```
# Assignment-1
