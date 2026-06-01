# Audi A1 Price Prediction using Machine Learning

This project focuses on predicting the prices of used Audi A1 cars based on their specifications (Year, Mileage, Engine volume, Transmission, and Fuel type). The project covers the entire workflow of a Machine Learning task, including data cleaning, data preprocessing, and hyperparameter tuning using Random Forest Regressor.

## 🚀 Key Features
* **Data Cleaning & Engineering:** Drop irrelevant columns, handle custom text formatting (e.g., removing 'L' from engine specifications), and rename messy column names.
* **Feature Encoding:** Categorical features (`Transmission` and `Fuel`) are safely encoded into numerical form using One-Hot Encoding (`pd.get_dummies`) while avoiding the dummy variable trap.
* **Hyperparameter Tuning:** Utilizing `GridSearchCV` with 5-fold cross-validation to search for the absolute best parameters (`n_estimators`, `max_depth`, `min_samples_split`) to optimize model performance.
* **Evaluation Matrix:** Evaluated based on Mean Absolute Error (MAE) to ensure the model makes reliable real-world price estimates.

## 🛠️ Tech Stack & Libraries
* **Python 3**
* **Pandas** (Data manipulation and cleaning)
* **Scikit-Learn** (Data splitting, Random Forest modeling, Grid Search CV, and MAE evaluation)

## 📁 Project Structure & Code Flow
The script executes the following sequential data pipeline:
1. **Load Data:** Imports the `Audi_A1_listings.csv` dataset.
2. **Data Cleansing:** Drops features like links, indexes, and ranks that don't contribute to the vehicle's actual value.
3. **Data Formatting:** Converts string types to proper numeric fields and renames complex column names (`Price(£)` and `Mileage(miles)`).
4. **Train-Test Split:** Separates features ($X$) and target variable ($y$), splitting the data into 80% training and 20% testing blocks.
5. **Optimization:** Initiates a grid search matrix over the Random Forest model to lock down the optimal estimator constraints.
6. **Results:** Outputs the predicted arrays alongside the final Optimized Mean Absolute Error (MAE).

## 📊 Sample Parameter Grid Investigated
```python
grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10]
}
