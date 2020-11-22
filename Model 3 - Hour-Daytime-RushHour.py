import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

data_path = r'data/clean_df.csv.gz'
df = pd.read_csv(data_path)

# Transform datetime variables
df['CRASH DATE'] = pd.to_datetime(df['CRASH DATE'])
df['CRASH TIME'] = pd.to_datetime(df['CRASH TIME'])

# Add target variable
df['CASUALTIES?'] = 0
mask = df['TOTAL PEDESTRIAN CASUALTIES'] != 0
df.loc[mask, 'CASUALTIES?'] = 1
df.loc[df['TOTAL PEDESTRIAN CASUALTIES'] != 1, ['TOTAL PEDESTRIAN CASUALTIES','CASUALTIES?']].sample    (5)

# Random Forest hyperparameters
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
    'max_iter':10_000
}

# KMeans hyperparameters
kmeans_params = {
    'n_clusters':52,
    'random_state':42
}

# Add K-means cluster features
clusters=52
kmeans = KMeans(**kmeans_params)
kmeans.fit(df[['LATITUDE','LONGITUDE']].values)
df['CLUSTERS'] = kmeans.labels_

# Add borough features
borough_dummies = pd.get_dummies(df['BOROUGH'], sparse=True)
cluster_dummies = pd.get_dummies(df['CLUSTERS'], prefix='CLUSTER', sparse=True)
pre_X = cluster_dummies.join(borough_dummies)

# Add year/month/season features
df['YEAR'] = df['CRASH DATE'].dt.year
df['MONTH'] = df['CRASH DATE'].dt.month

year_dummies = pd.get_dummies(df['YEAR'], sparse=True, prefix='YEAR')
month_dummies = pd.get_dummies(df['MONTH'], sparse=True)
season_dummies = pd.get_dummies(df['SEASON'], sparse=True)

df['HOUR'] = df['CRASH TIME'].dt.hour
hour_dummies = pd.get_dummies(df['HOUR'], sparse=True, prefix='HOUR')

pre_X = pre_X.join(df[['DURING DAYTIME','RUSH HOUR']]).join(hour_dummies)

# Train-test split
X = scipy.sparse.csr_matrix(pre_X)
y = df['CASUALTIES?']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Models
log_reg = LogisticRegression(**log_params)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
log_f1 = f1_score(y_test, y_pred)

rf_clf = RandomForestClassifier(**rf_params)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)
rf_f1 = f1_score(y_test, y_pred)

print(f'Logistic Regression F1: {log_f1}\nRandom Forest F1: {rf_f1}')