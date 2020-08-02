#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 10:53:54 2020

@author: anakromeiro
"""

import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the data
bmi_life_data = pd.read_csv("data.csv", delimiter = ',')

# Make and fit the linear regression model
bmi_life_model = LinearRegression()
bmi_life_model.fit(bmi_life_data[['BMI']], bmi_life_data[['Life expectancy']])

# Mak a prediction using the model
laos_life_exp = bmi_life_model.predict([[21.07931]])

print(laos_life_exp)

