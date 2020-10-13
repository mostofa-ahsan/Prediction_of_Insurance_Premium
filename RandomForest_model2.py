# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 23:15:21 2020

@author: mosto
"""


import pandas as pd


train= pd.read_csv('C:/Users/mosto/REPOSITORY/Prediction_of_Insurance_Premium/data/Train.csv', index_col=False)
test= pd.read_csv('C:/Users/mosto/REPOSITORY/Prediction_of_Insurance_Premium/data/Test.csv', index_col=False)
val= pd.read_csv('C:/Users/mosto/REPOSITORY/Prediction_of_Insurance_Premium/data/Validation.csv', index_col=False)


y_train = train["Monthly Premium Auto"]
y_test = test["Monthly Premium Auto"]
y_val = val["Monthly Premium Auto"]



drop_columns = ["Monthly Premium Auto", "Customer Lifetime Value", "Total Claim Amount", 
                "Response", "Renew Offer Type" ]

x_train = train.drop(columns = drop_columns)
x_test = test.drop(columns = drop_columns)

x_train = x_train.drop(columns = 'Activation_date')
x_test = x_test.drop(columns = 'Activation_date')


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42)

# fitting random regression model

rfModel = rf.fit(x_train, y_train)

# predicting y_values using test dataset. 

y_pred_r = rfModel.predict(x_test)

mae_r = mean_absolute_error(y_test, y_pred_r)
mse_r = mean_squared_error(y_test, y_pred_r)
print(f'Random Forest Regression mean absolute error {mae_r}')
print(f"Random Forest Regression mean squared error {mse_r}")

print(y_train[0]+y_train[1])