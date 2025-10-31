# Sport Car Price Prediction

## Project Description
This project aims to predict the price of sport cars using a dataset from Kaggle. The analysis involves exploring the dataset, preprocessing the data, training different regression models, and evaluating their performance.

## Data Source
The dataset used in this project is sourced from Kaggle:
[Sport Car Prices Dataset](https://www.kaggle.com/datasets/rkiattisak/sports-car-prices-dataset)

## Analysis and Results
In this notebook, we performed the following steps:
- Loaded and explored the dataset.
- Visualized key features like car make, year, and price.
- Handled missing values (if any were present).
- Preprocessed the data for model training (e.g., encoding categorical features, scaling numerical features).
- Trained several regression models (e.g., K-Nearest Neighbors, Decision Tree, XGBoost).
- Evaluated the models using cross-validation scores.

**Algorithms Used:**
- K-Nearest Neighbors Regressor (`KNeighborsRegressor`)
- Decision Tree Regressor (`DecisionTreeRegressor`)
- XGBoost Regressor (`XGBRegressor`)

**Results:**
The cross-validation scores for the trained models are as follows:
- KNeighbors cross-validation score: 91.84%
- Tree cross-validation score: 84.41%
- Xgboost cross-validation score: 89.76%

**Libraries Used:**
- pandas
- seaborn
- matplotlib.pyplot
- sklearn (StandardScaler, LabelEncoder, OneHotEncoder, SimpleImputer, ColumnTransformer, Pipeline, train_test_split, cross_val_score, DecisionTreeRegressor, KNeighborsRegressor, LogisticRegression)
- xgboost
- joblib

## How to Run the Notebook
(Provide instructions on how to set up the environment and run the notebook)

## Dependencies
(List any libraries or dependencies required to run the notebook)
