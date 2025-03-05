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