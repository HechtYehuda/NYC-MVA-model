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

# Import data
print('Importing crash data...')
data_path = r'data/clean_df.csv.gz'
df = pd.read_csv(data_path)
print('Done.')

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
    'max_depth':20,
    'n_estimators':15,
    'max_features':None,
    'n_jobs':15,
    'random_state':42,
    'verbose':1
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

# TF-IDF hyperparameters
tfidf_params = {
    'min_df':30,
    'max_df':0.9
}

# Add K-means cluster features
print('Adding K-means features...')
clusters=52
kmeans = KMeans(**kmeans_params)
kmeans.fit(df[['LATITUDE','LONGITUDE']].values)
df['CLUSTERS'] = kmeans.labels_
print('Done.')

# Add borough features
print('Adding borough features...')
borough_dummies = pd.get_dummies(df['BOROUGH'], sparse=True)
cluster_dummies = pd.get_dummies(df['CLUSTERS'], prefix='CLUSTER', sparse=True)
pre_X = cluster_dummies.join(borough_dummies)
print('Done.')

# Add year/month/season features
print('Adding date/time features...')
df['YEAR'] = df['CRASH DATE'].dt.year
df['MONTH'] = df['CRASH DATE'].dt.month

year_dummies = pd.get_dummies(df['YEAR'], sparse=True, prefix='YEAR')
month_dummies = pd.get_dummies(df['MONTH'], sparse=True)
season_dummies = pd.get_dummies(df['SEASON'], sparse=True)

df['HOUR'] = df['CRASH TIME'].dt.hour
hour_dummies = pd.get_dummies(df['HOUR'], sparse=True, prefix='HOUR')

pre_X = pre_X.join(df[['DURING DAYTIME','RUSH HOUR']]).join(hour_dummies)
print('Done.')

# Add weekday features
print('Adding weekday features...')
weekday_dummies = pd.get_dummies(df['WEEKDAY'], sparse=True)
pre_X = pre_X.join(weekday_dummies)
print('Done.')

# Add street name features
print('Adding street name features...')
df['STREET NAME IS NULL'] = df['ON STREET NAME'].isnull().astype('int')
df['ON STREET NAME'] = df['ON STREET NAME'].fillna('')

tfidf = TfidfVectorizer(**tfidf_params)
tfidf_df = pd.DataFrame.sparse.from_spmatrix(tfidf.fit_transform(df['ON STREET NAME']))
tfidf_pre = tfidf_df.add_prefix('ON_')
pre_X_dense = pre_X.join(tfidf_pre).join(df['STREET NAME IS NULL'])
pre_X = scipy.sparse.csr_matrix(pre_X)
print('Done.')

# Add off street name features
print('Adding off street name features...')
df['OFF STREET NAME IS NULL'] = df['OFF STREET NAME'].isnull().astype('int')
df['OFF STREET NAME'] = df['OFF STREET NAME'].fillna('')

tfidf = TfidfVectorizer(**tfidf_params)
tfidf_df = pd.DataFrame.sparse.from_spmatrix(tfidf.fit_transform(df['OFF STREET NAME']))
tfidf_pre = tfidf_df.add_prefix('OFF_')
off_dense = tfidf_pre.join(df['OFF STREET NAME IS NULL'])
off_sparse = scipy.sparse.csr_matrix(off_dense)
pre_X = scipy.sparse.hstack([pre_X, off_sparse])
print('Done.')

# Train-test split
X = scipy.sparse.csr_matrix(pre_X)
y = df['CASUALTIES?']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Models
print('Modeling logistic regression...')
log_reg = LogisticRegression(**log_params)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
log_f1 = f1_score(y_test, y_pred)

print('Modeling random forest classifier...')
rf_clf = RandomForestClassifier(**rf_params)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)
rf_f1 = f1_score(y_test, y_pred)
print('Done.')

# Results
print(f'Logistic Regression F1: {log_f1}\nRandom Forest F1: {rf_f1}')

cm = confusion_matrix(y_test, y_pred)
print(cm)
_ = sns.heatmap(cm)
_ = plt.xlabel('True casualties')
_ = plt.ylabel('Predicted casualties')
_ = plt.show()
