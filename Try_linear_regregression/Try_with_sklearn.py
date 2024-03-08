from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
# from utils import *

data=pd.read_csv("placement.csv")
x_train = data['CGPA'].values.reshape(-1, 1)
y_train = data['Packages'].values
lr_model=LinearRegression()
lr_model.fit(x_train,y_train)

y_predicted=lr_model.predict(x_train)
print(y_predicted)