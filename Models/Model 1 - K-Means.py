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
from sklearn.metrics import recall_score, confusion_matrix, make_scorer

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


# Fit K-Means
print('Fitting K-means clusters...')
counts = [7, 3, 4, 3, 5]
boroughs = ['MANHATTAN','QUEENS','BROOKLYN','STATEN ISLAND','BRONX']
for n, borough in zip(counts,boroughs):
    print(f'    Calculating {borough.title()} clusters...')
    
    borough_accidents = df[df['BOROUGH'] == borough]
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(borough_accidents[['LATITUDE','LONGITUDE']].values)
    
    df.loc[df['BOROUGH'] == borough, f'{borough} CLUSTERS'] = kmeans.labels_

print('Done.')


# Create dummies
print('Creating feature set...')
borough_dummies = pd.get_dummies(df['BOROUGH'], sparse=True)
borough_clusters = [borough+' CLUSTERS' for borough in boroughs]
cluster_dummies = pd.get_dummies(df[borough_clusters].fillna(''), prefix='CLUSTER', sparse=True)
pre_X = cluster_dummies.join(borough_dummies)

# Train-test split
X = scipy.sparse.csr_matrix(pre_X)
y = df['CASUALTIES?']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('Done.')

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

cm = confusion_matrix(y_test, y_test_pred, normalize='true')

_ = sns.heatmap(cm)
_ = plt.xlabel('True casualties')
_ = plt.ylabel('Predicted casualties')

_ = plt.show()

