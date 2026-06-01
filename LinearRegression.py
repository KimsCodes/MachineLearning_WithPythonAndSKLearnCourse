# -*- coding: utf-8 -*-
"""
Created on Fri May  1 08:42:25 2026

@author: kimle
"""

import pandas as pd
import plotly.express as px
import plotly.io as pio
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor

# ============================================================================================
# SETTINGS 
# ============================================================================================
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'
pio.renderers.default = "png"   # alternatively: "svg", force Plotly to render as static image inside Spyder



# ============================================================================================
# DATA 
# ============================================================================================
medical_df = pd.read_csv("insurance.csv")



# ============================================================================================
# LINEAR REGRESSION
# ============================================================================================
# smoker and age have strongest corr with charges

non_smoker_df = medical_df[medical_df.smoker == "no"]
smoker_df = medical_df[medical_df.smoker == "yes"]
#print(non_smoker_df.head())

# w for weight, b for bias
def estimate_charges(age, w, b):
    return w * age + b

w = 267.24     # model.coef_
b = -2091.42   # model.intercept_
ages = smoker_df.age
estimate_charges = estimate_charges(ages, w, b)
target = smoker_df.charges
#print(estimate_charges)

# ============================================================================================
# LOSS
# ============================================================================================
# the lower the loss, the better the model | loss of information

# RMSE - root mean square error ------------------------------------------------
# output: on average, each element in the predition differs from the actual value by the output rmse
def rmse(targets, predictions):
    return np.sqrt(np.mean(np.square(targets - predictions)))


# PLOT ------------------------------------------------------------------------
# plt.plot(ages, estimate_charges, 'r', alpha=0.9);
# plt.scatter(ages, target, s=8, alpha=0.8);
# plt.xlabel('Age');
# plt.ylabel('Charges');
# plt.legend(['Estimate', 'Actual']);


# ============================================================================================
# LINEAR REGRESSION
# ============================================================================================
# ordinary least squares: better for smaller datasets (few thousand)
# stochastic gradient descent: better for bigger datasets 

# # 1 Create inputs and targes
# inputs = non_smoker_df[['age']] # 2 brakets: needs to be 2 dimensional df, [[ creates list of coloumns (in this case filled with one coloumn)
# targets = non_smoker_df.charges # this can be 1D

# # 2 Create and train model
# model_lin = LinearRegression().fit(inputs, targets)

# # 3 Generate prediction
# pred = model_lin.predict(inputs)

# # 4 Compute loss to evaluate the model
# rmse_lin = rmse(targets, pred)
# print(rmse_lin)


# ============================================================================================
# SGD REGRESSOR
# ============================================================================================
# # 2 ----------- create and train model
# model_sgd = SGDRegressor().fit(inputs, targets)

# # 3 ----------- generate prediction
# pred_sgd = model_sgd.predict(inputs)

# # 4 -----------comput eloss to evaluate the model
# rmse_sgd = rmse(targets, pred_sgd)
# print(rmse_sgd)


# ============================================================================================
# MULTIPLE FEATURES
# ============================================================================================
inputs, targets = medical_df[['age', 'children', 'bmi']], medical_df['charges']

model = LinearRegression().fit(inputs, targets)

pred = model.predict(inputs)

loss = rmse(targets, pred)
print(loss)














# Quote from teacher: ML and also Deep learning is most of the time just glorified line fitting













