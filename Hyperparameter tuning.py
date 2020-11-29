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
max_k = {}
for current_borough in boroughs:
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
        print(f'# Clusters: {i}\n    F1 score: {log_recall}')
        recall_list.append(log_recall)
    _ = plt.figure(figsize=(10,10))
    _ = plt.plot(range(2,21), recall_list, 'k-')
    _ = plt.grid()
    _ = plt.xlabel('# Clusters', fontsize=14)
    _ = plt.ylabel('F1 Score', fontsize=14)
    _ = plt.title(f'{current_borough} F1 Cluster Analysis\n', fontsize=22)
    _ = plt.show()
    max_k[current_borough] = {
                        'K':recall_list.index(max(recall_list))+2,
                        'Score': max(recall_list)
            }
print(max_k)



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
#     print(f'# Clusters: {i}\n    F1 score: {log_recall}')
#     recall_list.append(log_recall)
# 
# # Plot F1 cluster analysis
# _ = plt.figure(figsize=(10,10))
# _ = plt.plot(range(2,101), recall_list, 'k-')
# _ = plt.grid()
# _ = plt.xlabel('# Clusters', fontsize=14)
# _ = plt.ylabel('F1 Score', fontsize=14)
# _ = plt.title('F1 Cluster Analysis\n', fontsize=22)
# _ = plt.show()
# plt.savefig('F1 Cluster Analysis')
# 
# Best K
n_cluster = recall_list.index(max(recall_list))+2
print(n_cluster, recall_list[n_cluster])

# Plot K-means clusters
kmeans = KMeans(n_clusters=n_cluster, random_state=42)
kmeans.fit(df[['LATITUDE','LONGITUDE']].values)

_ = plt.scatter(df['LATITUDE'], df['LONGITUDE'], alpha=0.4)
_ = plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1])

# Create K-means feature set
labels = pd.Series(kmeans.labels_)
pre_X = pd.get_dummies(labels, sparse=True)

# Split X and y
X = scipy.sparse.csr_matrix(pre_X)
y = df['CASUALTIES?']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest hyperparameter search
params = {
    'max_depth':[1,2,5,10,15],
    'n_estimators':[10,30,100],
    'class_weight': ['balanced'],
    'max_features':[None],
    'random_state': [42],
    'n_jobs':[-1],
    'verbose':[2]
}

cv = GridSearchCV(estimator=RandomForestClassifier(), param_grid=params, scoring=make_scorer(recall_score), n_jobs=-1)
cv.fit(X_train, y_train)

cv_results = pd.DataFrame(cv.cv_results_)
cv_results[['param_max_depth','param_n_estimators','mean_test_score','mean_fit_time']].sort_values(by='mean_test_score', ascending=False)

print(f'{cv.best_params_}\n{cv.best_score_}')

rf_clf = RandomForestClassifier(max_depth=15, n_estimators=100, class_weight='balanced', max_features=None, random_state=42, n_jobs=-1)
rf_clf.fit(X_train, y_train)

y_pred = rf_clf.predict(X_test)
recall_score(y_test, y_pred)
