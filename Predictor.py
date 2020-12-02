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

# New data input
df = pd.DataFrame()
df['CRASH DATE'] = input('Date (YYYY-MM-DD):\n')
df['CRASH TIME'] = input('Time (HH:MM:SS):\n')
df['BOROUGH'] = input('Borough:\n')
df['ON STREET NAME'] = input('Street:\n')
df['CROSS STREET NAME'] = input('Cross street:\n')

# Feature extraction
df['CRASH DATE'] = pd.to_datetime(df['CRASH DATE'])
df['CRASH TIME'] = pd.to_datetime(df['CRASH TIME'])
df['YEAR'] = df['CRASH DATE'].dt.year
df['MONTH'] = df['CRASH DATE'].dt.month
print(df)
df.loc[df['CRASH DATE'].dt.month.isin([3,4,5]) == True, 'SEASON'] = 'spring'
df.loc[df['CRASH DATE'].dt.month.isin([6,7,8]) == True, 'SEASON'] = 'summer'
df.loc[df['CRASH DATE'].dt.month.isin([9,10,11]) == True, 'SEASON'] = 'fall'
df.loc[df['CRASH DATE'].dt.month.isin([12,1,2]) == True, 'SEASON'] = 'winter'
df['HOUR'] = df['CRASH TIME'].dt.hour

print(df)
