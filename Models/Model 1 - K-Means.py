# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import scipy.stats as stats
import scipy.sparse
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, confusion_matrix, make_scorer
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

# Import data
print('Importing data...')
data_path = r'data/clean_df.csv.gz'
df = pd.read_csv(data_path)
print('Done.')

# Set target variable
print('Setting target variable...')
df['CASUALTIES?'] = 0
mask = df['TOTAL PEDESTRIAN CASUALTIES'] != 0
df.loc[mask, 'CASUALTIES?'] = 1
df.loc[df['TOTAL PEDESTRIAN CASUALTIES'] != 1, ['TOTAL PEDESTRIAN CASUALTIES','CASUALTIES?']].sample(5)
print('Done.')

# Random Forest hyperparameters
print('Setting hyperparameters...')
rf_params = {
    'class_weight':'balanced',
    'max_depth':15,
    'n_estimators':10,
    'max_features':None,
    'n_jobs':-1,
    'random_state':42
}

# Logistic Regression hyperparameters
log_params = {
    'class_weight':'balanced',
    'max_iter':10_000,
    'tol':0.001,
    'n_jobs':-1
}

# KMeans hyperparameters
kmeans_params = {
    'n_clusters':52,
    'random_state':42
}
print('Done.')

# Fit K-Means
print('Fitting K-means clusters...')
clusters=52
kmeans = KMeans(**kmeans_params)
kmeans.fit(df[['LATITUDE','LONGITUDE']].values)
print('Done.')

# Plot K clusters
_ = plt.scatter(df['LATITUDE'], df['LONGITUDE'], alpha=0.4)
_ = plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1])
_ = plt.show()

df['CLUSTERS'] = kmeans.labels_

# Create dummies
print('Creating feature set...')
borough_dummies = pd.get_dummies(df['BOROUGH'], sparse=True)
cluster_dummies = pd.get_dummies(df['CLUSTERS'], prefix='CLUSTER', sparse=True)
pre_X = cluster_dummies.join(borough_dummies)

X = scipy.sparse.csr_matrix(pre_X)
y = df['CASUALTIES?']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('Done.')

# Modeling
print('Creating logistic regression model...')
log_reg = LogisticRegression(**log_params)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
log_f1 = f1_score(y_test, y_pred)
print('Done.')

print('Creating random forest classifier model...') 
rf_clf = RandomForestClassifier(**rf_params)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)
rf_f1 = f1_score(y_test, y_pred)

print(f'Logistic Regression F1: {log_f1}\nRandom Forest F1: {rf_f1}')

cm = confusion_matrix(y_test, y_pred)
_ = sns.heatmap(cm)
_ = plt.xlabel('True casualties')
_ = plt.ylabel('Predicted casualties')

_ = plt.show()

