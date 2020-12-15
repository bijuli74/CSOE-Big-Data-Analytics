import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

cnames =  ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol', 'quality']
dataset = pd.read_csv('wineQualityRed_train.csv', sep=';', names=cnames, skiprows=1)

features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']
x = dataset[features]
y = dataset['quality']

x_train= x.values
y_train= y.values

dataset2 = pd.read_csv('wineQualityRed_test.csv', sep=';', names=cnames, skiprows=1)
x2 = dataset2[features]
y2 = dataset2['quality']
x_test = x2.values
y_test = y2.values


reg = LinearRegression()
reg.fit(x_train, y_train)

#prediction
y_pred = reg.predict(x_test)

#MSE calculation

mse = mean_absolute_error(y_test, y_pred)
print('Mean Squared Error: ', mse)


#Graph
plt.scatter(x_test, y_test, color='red')
plt.plot(x_test, y_pred, color='blue')
plt.title('Quality vs (Training Set)')
plt.xlabel('xxxxx')
plt.ylabel('Quality')
plt.show()


