import pandas as pd
import numpy as np
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
    'random_state':42,
    'verbose':1
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

# TF-IDF hyperparameters
count_params = {
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

# # Fit K-Means
# print('Fitting K-means clusters...')
# counts = [7, 3, 4, 3, 5]
# boroughs = ['MANHATTAN','QUEENS','BROOKLYN','STATEN ISLAND','BRONX']
# for n, borough in zip(counts,boroughs):
#     print(f'    Calculating {borough.title()} clusters...')
    
#     borough_accidents = df[df['BOROUGH'] == borough]
#     kmeans = KMeans(n_clusters=n, random_state=42)
#     kmeans.fit(borough_accidents[['LATITUDE','LONGITUDE']].values)
    
#     df.loc[df['BOROUGH'] == borough, f'{borough} CLUSTERS'] = kmeans.labels_

# print('Done.')

# # Create dummies
# print('Creating feature set...')
# borough_dummies = pd.get_dummies(df['BOROUGH'], sparse=True)
# borough_clusters = ['MANHATTAN CLUSTERS',
#                     'QUEENS CLUSTERS',
#                     'BROOKLYN CLUSTERS',
#                     'STATEN ISLAND CLUSTERS',
#                     'BRONX CLUSTERS']
# cluster_dummies = pd.get_dummies(df[borough_clusters].fillna(''), prefix='CLUSTER', sparse=True)
# pre_X = cluster_dummies.join(borough_dummies)

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

count = CountVectorizer(min_df=50, max_df=.5)
count_df = pd.DataFrame.sparse.from_spmatrix(count.fit_transform(df['ON STREET NAME']))
count_pre = count_df.add_prefix('ON')
pre_X = pre_X.join(count_pre).join(df['STREET NAME IS NULL'])
print('Done.')

# Train-test split
print('Splitting data...')
X = scipy.sparse.csc_matrix(pre_X)
y = df['CASUALTIES?']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
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
