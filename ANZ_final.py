# -*- coding: utf-8 -*-
"""
Created on Mon May 11 22:54:40 2020

@author: akhil
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_excel('ANZ synthesised transaction dataset.xls')

pay = dataset.loc[dataset['txn_description']=='PAY/SALARY']
salary = pay[['amount', 'customer_id']].groupby("customer_id").sum()
# Calculating for a year
salary['amount']*=4
salary.rename(columns = {'amount':'salary'}, inplace = True)

# Adding annual salary to the dataset
data = pd.merge(salary, dataset,  on='customer_id')

# Plotting salary vs age data
numeric = data[['salary','customer_id', 'gender', 'age']].groupby("customer_id").max()
# Creating colors list using the function below
def pltcolor(lst):
    cols=[]
    for l in lst:
        if l=="F":
            cols.append('red')
        elif l=="M":
            cols.append('blue')
    return cols
cols=pltcolor(numeric.gender)
plt.scatter(numeric.age, numeric.salary, c=cols)
plt.title('Annual Salary vs Age')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()


# Linear regression
# Preparing data for training
model_data = data.groupby("customer_id").mean()
X = model_data.iloc[:, 1:].values 
X = np.delete(X, 1, axis= 1)
y = (model_data.iloc[:, 0].values).astype(int)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Creating model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print("Accuracy of Linear model")
print(regressor.score(X_test, y_test))
print("We can see Accuracy is very low. So, ANZ should not use this model for predicting\n")

# Decision tree
data = data[['salary', "txn_description", "gender", "age", "merchant_state", "movement"]]
X = pd.get_dummies(data).iloc[:,1:]
y = (data.iloc[:, 0].values).astype(int)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Creating Model
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

print("Accuracy of Decision Tree model")
print(regressor.score(X_test, y_test))


