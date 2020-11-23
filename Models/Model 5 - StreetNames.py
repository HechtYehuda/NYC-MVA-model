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
print('Adding cluster features...')
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

# Add weekday features
weekday_dummies = pd.get_dummies(df['WEEKDAY'], sparse=True)
pre_X = pre_X.join(weekday_dummies)

# Add street name features
print('Adding street name features...')
df['STREET NAME IS NULL'] = df['ON STREET NAME'].isnull().astype('int')
df['ON STREET NAME'] = df['ON STREET NAME'].fillna('')

tfidf = TfidfVectorizer(min_df=50, max_df=.5)
tfidf_df = pd.DataFrame.sparse.from_spmatrix(tfidf.fit_transform(df['ON STREET NAME']))
tfidf_pre = tfidf_df.add_prefix('ON')
pre_X = pre_X.join(tfidf_pre).join(df['STREET NAME IS NULL'])
print('Done.')

# Train-test split
X = scipy.sparse.csc_matrix(pre_X)
y = df['CASUALTIES?']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modeling
print('Creating logistic regression model...')
log_reg = LogisticRegression(**log_params)
log_reg.fit(X_train, y_train)

y_train_pred = log_reg.predict(X_train)
y_test_pred = log_reg.predict(X_test)

log_train_f1 = f1_score(y_train, y_train_pred)
log_test_f1 = f1_score(y_test, y_test_pred)
print('Done.')

print('Creating random forest classifier model...') 
rf_clf = RandomForestClassifier(**rf_params)
rf_clf.fit(X_train, y_train)

y_train_pred = rf_clf.predict(X_train)
y_test_pred = rf_clf.predict(X_test)

rf_train_f1 = f1_score(y_train, y_train_pred)
rf_test_f1 = f1_score(y_test, y_test_pred)

print(f'Train scores:\n    Logistic Regression F1: {log_train_f1}\n    Random Forest F1: {rf_train_f1}')
print(f'Test scores:\n    Logistic Regression F1: {log_test_f1}\n    Random Forest F1: {rf_test_f1}')

cm = confusion_matrix(y_test, y_test_pred)
_ = sns.heatmap(cm)
_ = plt.xlabel('True casualties')
_ = plt.ylabel('Predicted casualties')

_ = plt.show()