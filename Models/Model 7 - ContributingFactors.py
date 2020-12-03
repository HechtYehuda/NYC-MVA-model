import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import scipy.stats as stats
import scipy.sparse
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import recall_score, confusion_matrix, make_scorer

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
df.loc[df['TOTAL PEDESTRIAN CASUALTIES'] != 1, ['TOTAL PEDESTRIAN CASUALTIES','CASUALTIES?']].sample(5)

# Random Forest hyperparameters
params_path = r'Predictor tools/rf_params.pickle'
with open(params_path, 'rb') as file:
    rf_params = pickle.load(file)

# Logistic Regression hyperparameters
log_params = {
    'class_weight':'balanced',
    'max_iter':10_000,
    'tol':0.001,
    'n_jobs':-1
}

# TF-IDF hyperparameters
count_params = {
    'min_df':30,
    'max_df':0.9
}

# Add K-means cluster features
print('Adding K-means features...')
params_path = r'Predictor tools/k_clusters.pickle'
with open(params_path, 'rb') as file:
    max_k = pickle.load(file)

boroughs = ['MANHATTAN','BROOKLYN','STATEN ISLAND','QUEENS','BRONX']
k_clusters = []
for i in max_k:
    k_clusters.append(max_k[i]['K'])
for n, borough in zip(k_clusters,boroughs):
    print(f'    Calculating {borough.title()} clusters...')
    
    borough_accidents = df[df['BOROUGH'] == borough]
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(borough_accidents[['LATITUDE','LONGITUDE']].values)
    
    df.loc[df['BOROUGH'] == borough, f'{borough} CLUSTERS'] = kmeans.labels_
print('Done.')

# Add borough features
print('Adding borough features...')
borough_dummies = pd.get_dummies(df['BOROUGH'], sparse=True)
borough_clusters = [borough+' CLUSTERS' for borough in boroughs]
cluster_dummies = pd.get_dummies(df[borough_clusters].fillna(''), prefix='CLUSTER', sparse=True)
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

on_count = CountVectorizer(**count_params)
on_count_df = pd.DataFrame.sparse.from_spmatrix(on_count.fit_transform(df['ON STREET NAME']))
on_count_pre = on_count_df.add_prefix('ON_')
pre_X_dense = pre_X.join(on_count_pre).join(df['STREET NAME IS NULL'])
pre_X = scipy.sparse.csr_matrix(pre_X)
print('Done.')

# Add off street name features
print('Adding cross street name features...')
df['CROSS STREET NAME IS NULL'] = df['CROSS STREET NAME'].isnull().astype('int')
df['CROSS STREET NAME'] = df['CROSS STREET NAME'].fillna('')

cross_count = CountVectorizer(**count_params)
cross_count_df = pd.DataFrame.sparse.from_spmatrix(cross_count.fit_transform(df['CROSS STREET NAME']))
cross_count_pre = cross_count_df.add_prefix('CROSS_')
off_dense = cross_count_pre.join(df['CROSS STREET NAME IS NULL'])
off_sparse = scipy.sparse.csr_matrix(off_dense)
pre_X = scipy.sparse.hstack([pre_X, off_sparse])
print('Done.')

# Contributing factors
print('Adding contributing factors...')
factors_dummies = pd.get_dummies(df['CONTRIBUTING FACTOR VEHICLE 1'], sparse=True)
factors_sparse = scipy.sparse.csr_matrix(factors_dummies)
pre_X = scipy.sparse.hstack([pre_X, factors_sparse])
print('Done.')

# Train-test split
X = scipy.sparse.csr_matrix(pre_X)
y = df['CASUALTIES?']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modeling
print('Creating logistic regression model...')
log_reg = LogisticRegression(**log_params)
log_reg.fit(X_train, y_train)

y_train_pred = log_reg.predict(X_train)
y_test_pred = log_reg.predict(X_test)

log_train_recall = recall_score(y_train, y_train_pred)
log_test_recall = recall_score(y_test, y_test_pred)
print('Done.')

print('Creating random forest classifier model...') 
rf_clf = RandomForestClassifier(**rf_params)
rf_clf.fit(X_train, y_train)

y_train_pred = rf_clf.predict(X_train)
y_test_pred = rf_clf.predict(X_test)

rf_train_recall = recall_score(y_train, y_train_pred)
rf_test_recall = recall_score(y_test, y_test_pred)

print(f'Train scores:\n    Logistic Regression Recall: {log_train_recall}\n    Random Forest Recall: {rf_train_recall}')
print(f'Test scores:\n    Logistic Regression Recall: {log_test_recall}\n    Random Forest Recall: {rf_test_recall}')

def fp_rate(y_test, y_pred):
    tn, fp, fn, tp  = confusion_matrix(y_test, y_pred).ravel()
    return fp / (fp + tn)
    
def fn_rate(y_test, y_pred):
    tn, fp, fn, tp  = confusion_matrix(y_test, y_pred).ravel()
    return fn / (tp + fn)

print('False positive rate: ', fp_rate(y_test, y_test_pred) 
print('False negative rate: ', fn_rate(y_test, y_test_pred))

cm = confusion_matrix(y_test, y_test_pred)
_ = sns.heatmap(cm)
_ = plt.xlabel('True casualties')
_ = plt.ylabel('Predicted casualties')

_ = plt.show()
