# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 07:30:57 2026

@author: kimle
"""

import pandas as pd
import plotly.express as px
import plotly.io as pio
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

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
# EXPLORE DATA 
# ============================================================================================

# AGE ---------------------------------------------------------------------------------
# fig_age = px.histogram(
#     medical_df,
#     x='age',
#     marginal='box',
#     nbins=47,
#     title='Distribution of Age'
# )
# fig_age.update_layout(bargap=0.1)
# fig_age.show()


# fig = px.scatter(medical_df,
#                  x='bmi',
#                  y='charges',
#                  color='smoker',
#                  opacity=0.8,
#                  title='BMI vs. Charges')
# fig.show()


# BMI ---------------------------------------------------------------------------------
# fig_bmi = px.histogram(
#     medical_df,
#     x='bmi',
#     marginal='box',
#     title='Distribution of BMI'
# )
# fig_bmi.update_layout(bargap=0.1)
# fig_bmi.show()

# CHARGES ---------------------------------------------------------------------------------
# fig_charges = px.histogram(
#     medical_df,
#     x='smoker',
#     color='sex',
#     color_discrete_sequence=['lightpink', 'lightblue'],
#     title='Distribution of Smoker'
# )
# fig_charges.update_layout(bargap=0.1)
# fig_charges.show()



# ============================================================================================
# CORRELATION
# ============================================================================================

corr = medical_df.charges.corr(medical_df.bmi)
print(corr)



# ============================================================================================
# CORRELATION
# ============================================================================================










