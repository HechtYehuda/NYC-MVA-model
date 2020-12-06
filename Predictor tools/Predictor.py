import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy.sparse
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import recall_score
from astral.sun import sun

# API key
with open('api_key.txt', 'r') as file:
    key = file.read()
print(key)
### New data input
df = pd.DataFrame()
df.loc[0,'CRASH DATE'] = input('Date (YYYY-MM-DD):\n')
df.loc[0,'CRASH TIME'] = input('Time (HH:MM:SS):\n')
df.loc[0,'BOROUGH'] = input('Borough:\n')
df.loc[0,'ON STREET NAME'] = input('Street:\n')
df.loc[0,'CROSS STREET NAME'] = input('Cross street:\n')
print(df)

# Feature extraction
df['CRASH DATE'] = pd.to_datetime(df['CRASH DATE'])
df['CRASH TIME'] = pd.to_datetime(df['CRASH TIME'])
df['YEAR'] = df['CRASH DATE'].dt.year
df['MONTH'] = df['CRASH DATE'].dt.month
df.loc[df['CRASH DATE'].dt.month.isin([3,4,5]) == True, 'spring'] = 1
df.loc[df['CRASH DATE'].dt.month.isin([6,7,8]) == True, 'summer'] = 1
df.loc[df['CRASH DATE'].dt.month.isin([9,10,11]) == True, 'fall'] = 1
df.loc[df['CRASH DATE'].dt.month.isin([12,1,2]) == True, 'winter'] = 1
df['HOUR'] = df['CRASH TIME'].dt.hour
df = df.join(pd.get_dummies(df['BOROUGH'])).drop('BOROUGH', axis=1)
print(df.info())

