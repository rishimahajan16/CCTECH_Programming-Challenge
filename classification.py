# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 18:48:08 2020

@author: Rishi GOPAL MAHAJAN
"""

#Import Libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing

#Read the dataset
df=pd.read_csv('challenge2_dataset.csv')
df.head()

#Read the Prediction data
df1=pd.read_csv('challenge2_prediction.csv')
df1.head()

#Eliminate NAN values of column df 
mean=df['1'].mean()
df['1'].replace(np.nan,mean, inplace=True)
print("number of NaN values for the 1 :",df['1'].isnull().sum())

mean=df['2'].mean()
df['2'].replace(np.nan,mean, inplace=True)
print("number of NaN values for the 2 :",df['2'].isnull().sum())

mean=df['3'].mean()
df['3'].replace(np.nan,mean, inplace=True)
print("number of NaN values for the 3 :",df['3'].isnull().sum())

#Eliminate NAN values of df1
mean=df1['1'].mean()
df1['1'].replace(np.nan,mean, inplace=True)
print("number of NaN values for the 1 :",df['1'].isnull().sum())

mean=df1['2'].mean()
df1['2'].replace(np.nan,mean, inplace=True)
print("number of NaN values for the 2 :",df1['2'].isnull().sum())

mean=df1['3'].mean()
df1['3'].replace(np.nan,mean, inplace=True)
print("number of NaN values for the 3 :",df1['3'].isnull().sum())

#Convert target in dummies
dummy=pd.get_dummies(df['Target'], drop_first=True)
df2=pd.concat((df,dummy),axis=1)
df2=df2.drop(['Target'], axis=1)
df2.head()

df2.rename(columns = {df.columns[3] :'Target'})

#Classifications
x=df2.iloc[:,:-1].values
y=df2.iloc[:,-1].values
x_pred=df1.iloc[:,:].values

##Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(x, y)
predictions = model.predict(x_pred)

#Add knn_yhat column to dataset
y_hats_df = pd.DataFrame(data =predictions, columns = ['Target'], index = df1.index.copy())
Output = pd.merge(df1, y_hats_df, how = 'left', left_index = True, right_index = True)
Output.to_csv(r'C:\Users\PARAG GOPAL MAHAJAN\Desktop\Expected Targets.csv',index=False)
    