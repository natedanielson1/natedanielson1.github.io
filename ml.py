import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the stock data
data = pd.read_csv('QQQ 5year.csv')

# Convert dates into numerical values for regression
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = data['Date'].map(pd.Timestamp.toordinal)

# Our independent variable (X) will be the date
# Our dependent variable (Y) will be the closing price
X = data['Date'].values.reshape(-1, 1)
Y = data['Close'].values

# Splitting the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Creating a Linear Regression model
model = LinearRegression()

# Fitting the model with the training data
model.fit(X_train, Y_train)

# Making predictions
Y_pred = model.predict(X_test)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_test, Y_pred, color='blue')
plt.title('Stock Price Prediction using Linear Regression')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.show()
