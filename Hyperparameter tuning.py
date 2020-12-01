# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import scipy.stats as stats
import scipy.sparse
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import recall_score, confusion_matrix, make_scorer

# Random Forest hyperparameters 
params = {
    'max_depth':[1,2,5,10,15],
    'n_estimators':[10,30,100],
    'class_weight': ['balanced'],
    'max_features':[None],
    'random_state': [42],
    'n_jobs':[-1],
    'verbose':[2]
}

# Import data
print('Importing data...')
data_path = r'data/clean_df.csv.gz'
df = pd.read_csv(data_path)
print('Done.')

print('Engineering features...')
df['CRASH DATE'] = pd.to_datetime(df['CRASH DATE'])
df['CRASH TIME'] = pd.to_datetime(df['CRASH TIME'])

df['CASUALTIES?'] = 0
mask = df['TOTAL PEDESTRIAN CASUALTIES'] != 0
df.loc[mask, 'CASUALTIES?'] = 1
df.loc[df['TOTAL PEDESTRIAN CASUALTIES'] != 1, ['TOTAL PEDESTRIAN CASUALTIES','CASUALTIES?']].sample(5)
print('Done.')

# K Means analysis
boroughs = ['MANHATTAN','BROOKLYN','STATEN ISLAND','QUEENS','BRONX']
subplots = [231,232,233,234,235]
plt.figure()
max_k = {}
for space, current_borough in zip(subplots, boroughs):
    print(f'{current_borough.title()} K-Means analysis')
    borough = df[df['BOROUGH'] == current_borough]
    recall_list = []
    for i in range(2,21):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(borough[['LATITUDE','LONGITUDE']].values)
        df_clusters = pd.Series(kmeans.labels_)
        cluster_dummies = pd.get_dummies(df_clusters)
        X = scipy.sparse.csr_matrix(cluster_dummies)
        y = borough['CASUALTIES?']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        log_reg = LogisticRegression(class_weight='balanced', max_iter=10_000)
        log_reg.fit(X_train, y_train)
        y_pred = log_reg.predict(X_test)
        log_recall = recall_score(y_test, y_pred)
        print(f'# Clusters: {i}\n    Recall score: {log_recall}')
        recall_list.append(log_recall)
    plt.subplot(space)
    plt.plot(range(2,21), recall_list, 'k-')
    plt.grid()
    plt.xlabel(f'{current_borough.title()} Clusters', fontsize=12)
    plt.ylabel('Recall Score', fontsize=12)
    plt.xticks(range(2,21), rotation=60, ha='right')
    max_k[current_borough] = {
                        'K':recall_list.index(max(recall_list))+2,
                        'Score': max(recall_list)
            }

plt.show()
plt.savefig('K-Means tuning.png')
for i in max_k:
    print(f'{i}\n    {max_k[i]}')

# Fit K-Means
print('Fitting K-means clusters...')
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
# print('Beginning K-Means anlysis.')
# recall_list = []
# for i in range(2,101):
#     kmeans = KMeans(n_clusters=i, random_state=42)
#     kmeans.fit(df[['LATITUDE','LONGITUDE']].values)
#     df_clusters = pd.Series(kmeans.labels_)
#     cluster_dummies = pd.get_dummies(df_clusters)
#     X = scipy.sparse.csr_matrix(cluster_dummies)
#     y = df['CASUALTIES?']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#     log_reg = LogisticRegression(class_weight='balanced', max_iter=10_000)
#     log_reg.fit(X_train, y_train)
#     y_pred = log_reg.predict(X_test)
#     log_recall = recall_score(y_test, y_pred)
#     print(f'# Clusters: {i}\n    Recall score: {log_recall}')
#     recall_list.append(log_recall)
# 
# # Plot Recall cluster analysis
# _ = plt.figure(figsize=(10,10))
# _ = plt.plot(range(2,101), recall_list, 'k-')
# _ = plt.grid()
# _ = plt.xlabel('# Clusters', fontsize=14)
# _ = plt.ylabel('Recall Score', fontsize=14)
# _ = plt.title('Recall Cluster Analysis\n', fontsize=22)
# _ = plt.show()
# plt.savefig('Recall Cluster Analysis')
# 

# Create feature set
print('Creating feature set...')
borough_dummies = pd.get_dummies(df['BOROUGH'], sparse=True)
borough_clusters = [borough+' CLUSTERS' for borough in boroughs]
cluster_dummies = pd.get_dummies(df[borough_clusters].fillna(''), prefix='CLUSTER', sparse=True)
pre_X = cluster_dummies.join(borough_dummies)
print('Done.')

# Split X and y
print('Splitting data...')
X = scipy.sparse.csr_matrix(pre_X)
y = df['CASUALTIES?']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('Done.')

# Random Forest grid search
cv = GridSearchCV(estimator=RandomForestClassifier(), param_grid=params, scoring=make_scorer(recall_score), n_jobs=-1, return_train_score=True)
cv.fit(X_train, y_train)

cv_results = pd.DataFrame(cv.cv_results_)
print(cv_results[['param_max_depth','param_n_estimators','mean_train_score','mean_test_score','mean_fit_time']].sort_values(by='mean_test_score', ascending=False))

print(f'{cv.best_params_}\n{cv.best_score_}')

rf_clf = RandomForestClassifier(**cv.best_params_)
rf_clf.fit(X_train, y_train)

y_pred = rf_clf.predict(X_test)
print(cv.best_params_)
print(recall_score(y_test, y_pred))
