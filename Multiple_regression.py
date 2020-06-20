# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 18:46:39 2020

@author: Rishi GOPAL MAHAJAN
"""

#Problem Statement
# To train a ML model using the dataset.csv file and predict the target for the features of prediction.csv
#Add the predicted target column to the prediction.csv file with the column header "target"

#Import Libraries
import pandas as pd
import numpy as np

#Read the dataset
df=pd.read_csv('dataset.csv')
df1=pd.read_csv('prediction_reg.csv')

#Defininfg x and y
y=df["target"]
features=["0","1","2","3","4","5","6","7","8"]
x=df[features]
x_pred=df1[features]

#Multiple Regression model
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(x,y)
target=lm.predict(x_pred)

#Check R^2 value
print('R^2=',lm.score(x,y))

#Saving the target results
y_hats_df = pd.DataFrame(data = target, columns = ['target'], index = df1.index.copy())
Output = pd.merge(df1, y_hats_df, how = 'left', left_index = True, right_index = True)
Output.to_csv(r'C:\Users\PARAG GOPAL MAHAJAN\Desktop\reg_targets.csv',index=False)