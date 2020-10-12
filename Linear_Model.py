# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 22:33:25 2020

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



error = y_train.mean()-y_train


mean=y_train.mean()
mean_absolute_error = error.abs().mean()
mean_absolute_error
print(f'By guessing, our insurance premium would be ${round(mean, 2)} \nand we would be off by ${round(mean_absolute_error, 2)}')



from sklearn.linear_model import LinearRegression

lr = LinearRegression()


lrModel = lr.fit(x_train, y_train)
y_pred = lrModel.predict(x_test)



