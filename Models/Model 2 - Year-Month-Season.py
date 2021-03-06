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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import recall_score, confusion_matrix, make_scorer

# Import data
print('Importing data...')
data_path = r'data/clean_df.csv.gz'
df = pd.read_csv(data_path)

# Transform to datetime
df['CRASH DATE'] = pd.to_datetime(df['CRASH DATE'])
df['CRASH TIME'] = pd.to_datetime(df['CRASH TIME'])
print('Done.')

# Add target variable
print('Setting target...')
df['CASUALTIES?'] = 0
mask = df['TOTAL PEDESTRIAN CASUALTIES'] != 0
df.loc[mask, 'CASUALTIES?'] = 1
df.loc[df['TOTAL PEDESTRIAN CASUALTIES'] != 1, ['TOTAL PEDESTRIAN CASUALTIES','CASUALTIES?']].sample(5)
print('Done.')

# Random Forest hyperparameters
params_path = r'Predictor tools/rf_params.pickle'
with open(params_path, 'rb') as file:
    rf_params = pickle.load(file)

# Logistic Regression hyperparameters
log_params = {
    'class_weight':'balanced',
    'max_iter':10_000
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


# Create dummies
borough_dummies = pd.get_dummies(df['BOROUGH'], sparse=True)
borough_clusters = [borough+' CLUSTERS' for borough in boroughs]
cluster_dummies = pd.get_dummies(df[borough_clusters].fillna(''), prefix='CLUSTER', sparse=True)
pre_X = cluster_dummies.join(borough_dummies)

# Add year/month/season features
print('Adding date/time features...')
df['YEAR'] = df['CRASH DATE'].dt.year
df['MONTH'] = df['CRASH DATE'].dt.month

year_dummies = pd.get_dummies(df['YEAR'], sparse=True, prefix='YEAR')
month_dummies = pd.get_dummies(df['MONTH'], sparse=True)
season_dummies = pd.get_dummies(df['SEASON'], sparse=True)
month_year_dummies = pd.get_dummies(df['YEAR-MONTH'], sparse=True)

pre_X = pre_X.join(year_dummies).join(season_dummies).join(month_dummies).join(month_year_dummies)
print('Done.')

# Train-test split
print('Splitting data...')
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

sns.heatmap(cm)
plt.xlabel('True casualties')
plt.ylabel('Predicted casualties')

plt.savefig('Image resources/Model 2 confusion matrix.svg')
