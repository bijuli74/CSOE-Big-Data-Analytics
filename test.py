import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report, accuracy_score, f1_score, recall_score, precision_score
from sklearn.utils import multiclass
from sklearn.preprocessing import StandardScaler, LabelEncoder, binarize
from sklearn.decomposition import PCA

from prettytable import PrettyTable

cnames =  ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol', 'quality']
df = pd.read_csv('wineQualityRed_train.csv', sep=';', names=cnames, skiprows=1)

#binning the dataset TRAINING
df['quality'] = pd.cut(df['quality'], bins = [2, 7, 10], labels =['bad', 'good'], right = False)
enc = LabelEncoder()
df['quality'] = enc.fit_transform(df['quality'])

features = cnames[:-1]

x_train= df[features].values
y_train= df['quality'].values

df2 = pd.read_csv('wineQualityRed_test.csv', sep=';', names=cnames, skiprows=1)
x_test = df2[features].values
y_test = df2[features].values
#binning the dataset TESTING
df2['quality'] = pd.cut(df2['quality'], bins = [2, 7, 10], labels =['bad', 'good'], right = False)
enc2 = LabelEncoder()
df2['quality'] = enc2.fit_transform(df2['quality'])