import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score as cvs, KFold
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xg 
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import GridSearchCV 
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("GameData.csv")

datatypes = data.dtypes
print(datatypes)

#Feature Engineering
print(data.isnull().sum())
print(data.duplicated().sum())

df = data.dropna()

#Removing Outlier
Q1 = df["Global_Sales"].quantile(0.25)
Q3 = df["Global_Sales"].quantile(0.75)
IQR = Q3 - Q1
df = df[(df["Global_Sales"] >= Q1 - 1.5 * IQR) & (df["Global_Sales"] <= Q3 + 1.5 * IQR)]

print(df)

X = df.drop(columns=["Global_Sales", "Name", "Rank"])
y = df["Global_Sales"]

label_encoders = {}

for col in ['Platform', 'Genre', 'Publisher']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le  
    
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Linear Regression Model
RegModel = LinearRegression()
RegModel.fit(x_train, y_train)
y_pred1 = RegModel.predict(x_test)
print("MAE:", mean_absolute_error(y_test, y_pred1))
print("Linear Regression Model - Score:",RegModel.score(x_test, y_test))

#Random Forest Regression Model
RFmodel = RandomForestRegressor(n_estimators=100, random_state=42)
RFmodel.fit(x_train, y_train)

y_pred2 = RFmodel.predict(x_test)
print("MAE:", mean_absolute_error(y_test, y_pred2))
print("Random Forest Regression Model - Score:", RFmodel.score(x_test, y_test))

#XGB Regression Model
XGReg = xg.XGBRegressor(objective = 'reg:linear', n_estimator = 10, seed = 123)
XGReg.fit(x_train, y_train) 
y_pred3 = XGReg.predict(x_test)
print("MAE:", mean_absolute_error(y_test, y_pred3))
print("XGB Regression Model - Score:",XGReg.score(x_test, y_test))

#SV Regression Model
SVReg = SVR(C=1.0, epsilon = 0.2)
SVReg.fit(x_train, y_train)
y_pred4 = SVReg.predict(x_test)
print("MAE:", mean_absolute_error(y_test, y_pred4))
print("Support Vector Regression Model - Score:",SVReg.score(x_test, y_test))

#Model Optimization of SVR due to low score
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']} 

grid_search = GridSearchCV(SVReg, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=2)
grid_search.fit(x_train, y_train)

# Get best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Use the best model
best_model = grid_search.best_estimator_

y_pred_tuned = best_model.predict(x_test)
# Evaluate model performance
mse_tuned = mean_squared_error(y_test, y_pred_tuned)
print(f"Tuned Model - Mean Squared Error (MSE): {mse_tuned:.4f}")
print("Tuned Support Vector Regression Model - Score:",best_model.score(x_test, y_test))

# Saving the model
import joblib

# Save the trained model
joblib.dump(XGReg, "E:\AI Projects\Games Sales Prediction\game_sales_model.pkl")
joblib.dump(label_encoders, "E:\AI Projects\Games Sales Prediction\label_encoders.pkl")

print("Model saved successfully!")