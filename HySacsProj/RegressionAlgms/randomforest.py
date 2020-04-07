from numpy import array, hstack, math
from numpy.random import uniform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

df_ip = pd.read_csv("data/swir_ds.csv")
print(df_ip.iloc[:,1:])
df_out = pd.read_csv("data/tgt_sugar.csv", float_precision='round_trip')
print(df_out)
result = pd.merge(df_ip,
                  df_out[['Barcode','Print Info','Variety ID','Glucose','Fructose','Sucrose','Raffinose','1-Kestose','Maltose','Nystose','1,1,1-Kestopentaose','Total Fructan']],
                  left_on='Img_name',
                  right_on='Barcode',
                  how='inner')
print(result)

# Select the input features
X = result.iloc[:, 3:259].values
print(X)

#Select single target var
y = result.iloc[:, 262:].values
print(y);



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)

#Fitting Random Forest Regression model
regressor = RandomForestRegressor(n_estimators = 150, random_state = 0, criterion="mae")
params = {"n_estimators":[90, 120]}
clf = GridSearchCV(regressor, n_jobs=-1,  param_grid=params, cv=10, verbose=10)
clf.fit(X_train, y_train)
s = pickle.dumps(clf)

# #Calc mse for Training data
# y_true = y
# y_pred = clf.predict(X)
# print(mean_squared_error(y_true, y_pred)) #Train error = 0.01248 (mse)


def mse_score(tgt, mdl, iput):
    y_pred = mdl.predict(iput)
    mse = mean_squared_error(tgt, y_pred)  #Train error = 0.01248 (mse)
    return mse

mse_train = mse_score(tgt = y_train, mdl=clf, iput=X_train)  #mse_test =
mse_test = mse_score(tgt = y_test, mdl=clf, iput=X_test)   #mse_test = 0.8200
