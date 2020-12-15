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
df_new = df.iloc[:-1]
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

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#correlations btw 11 attributes 
correlations = df_new.corr()
# Plot figsize
fig, ax = plt.subplots(figsize=(10, 10))
colormap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(correlations, cmap=colormap, annot=True, fmt=".2f")
ax.set_xticklabels(
    features,
    rotation=45,
    horizontalalignment='right'
)
ax.set_yticklabels(features)
plt.show()


#PCA analysis
pcaA = PCA(n_components = 7)
# PCA(copy=True, iterated_power='auto', n_components=None,
# random_state=None,
#  svd_solver='auto', tol=0.0, whiten=False)
redwine_7_training = pcaA.fit_transform(x_train)
redwine_7_testing = pcaA.transform(x_test)


models  = []
models.append(('SupportVectorClassifier', SVC(C = 1.3, gamma =  1.3, kernel= 'rbf')))
models.append(('GaussianNB', GaussianNB()))
models.append(('LogisticRegression', LogisticRegression()))
models.append(('LinearRegression', LinearRegression()))

results = []
names = []
scoring = 'accuracy'
target_names = ['Bad Quality', 'Good Quality']
# warnings.filterwarnings('ignore')
ptbl = PrettyTable()
ptbl.field_names = ["Model", "Precision", "Recall", "F1Score"]

for name, model in models:
       temp_model = model
       temp_model.fit(redwine_7_training, y_train)
       y_pred_pca = temp_model.predict(redwine_7_testing)
       print(classification_report(y_test, y_pred_pca, target_names=target_names))
       acc_score = accuracy_score(y_test, y_pred_pca)
       results.append(acc_score)
       print('Accuracy score',acc_score*100)
       ptbl.add_row([name, precision_score(y_test, y_pred_pca, average = 'weighted'),
           recall_score(y_test, y_pred_pca, average = 'weighted'), f1_score(y_test, y_pred_pca, average = 'weighted')])
       print(ptbl)
       print('\n')

#b)


pcaB = PCA(n_components=4)
redwine_4_training = pcaB.fit_transform(x_train)
redwine_4_testing = pcaB.transform(x_test)


models_b  = []
models_b.append(('SupportVectorClassifier', SVC(C = 1.3, gamma =  1.3, kernel= 'rbf')))
models_b.append(('GaussianNB', GaussianNB()))
models_b.append(('LogisticRegression', LogisticRegression()))
models_b.append(('LinearRegression', LinearRegression()))

results_b = []
names_b = []
scoring_b = 'accuracy'
target_names_b = ['Bad Quality', 'Good Quality']
# warnings.filterwarnings('ignore')
ptbl_b = PrettyTable()
ptbl.field_names_b = ["Model", "Precision", "Recall", "F1Score"]

for name_b, model_b in models_b:
       temp_model_b= model_b
       temp_model_b.fit(redwine_4_training, y_train)
       y_pred_pca_b = temp_model_b.predict(redwine_4_testing)
       print(classification_report(y_test, y_pred_pca_b, target_names=target_names_b))
       acc_score_b = accuracy_score(y_test, y_pred_pca_b)
       results.append(acc_score_b)
       print('Accuracy score',acc_score_b*100)
       ptbl_b.add_row([name_b, precision_score(y_test, y_pred_pca_b, average = 'weighted'),
           recall_score(y_test, y_pred_pca_b, average = 'weighted'), f1_score(y_test, y_pred_pca_b, average = 'weighted')])
       print(ptbl_b)
       print('\n')
