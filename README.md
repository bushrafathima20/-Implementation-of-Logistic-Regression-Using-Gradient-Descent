# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and separate input and output variables.

2. Split the data into training and testing sets.

3. Train the linear regression model using the training data.

4. Predict the output for the test data and evaluate the results.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: BUSHRA FATHIMA I
RegisterNumber: 212225040051
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Placement_Data (1).csv")

data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})

X = data[['ssc_p', 'mba_p']].values
y = data['status'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

m = len(y)
X = np.c_[np.ones(m), X]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, theta):
    h = sigmoid(X @ theta)
    return (-1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))

theta = np.zeros(X.shape[1])
alpha = 0.1
cost_history = []

for i in range(500):
    z = X @ theta
    h = sigmoid(z)
    gradient = (1/m) * X.T @ (h - y)
    theta = theta - alpha * gradient
    
    cost = cost_function(X, y, theta)
    cost_history.append(cost)

y_pred = (sigmoid(X @ theta) >= 0.5).astype(int)

accuracy = np.mean(y_pred == y) * 100
print("Weights:", theta)
print("Accuracy:", accuracy, "%")

plt.figure()
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Logistic Regression using Gradient Descent")
plt.show()
```

## Output:
<img width="757" height="634" alt="Screenshot 2026-04-29 092611" src="https://github.com/user-attachments/assets/86f4a72c-9f03-44bd-b1d9-8324f2119d38" />

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

