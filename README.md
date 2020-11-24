# NYC MVA model

A model to predict pedestrian casualties based on accidents in NYC between 2012 and 2020.

The _Models_ folder contains the Python code to run each model. Data to be added to a DropBox and linked shortly.

F1 scores were used to calculate model accuracy, owing to the imbalanced nature of the data. LR = Logistic Regression; RF = Random Forest.

| Model # | Predictors | LR Train | RF Train | LR Test | RF Test |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | K-means clusters \| Boroughs | 0.166508 | 0.166507 | 0.166727 | 0.168079 |
| 2 | K-means clusters \| Boroughs \| Years \| Months \| Seasons | 0.167805 | 0.175968 | 0.167534 | 0.168375 |
| 3 | K-means clusters \| Boroughs \| Years \| Months \| Seasons \| Hour of day \| Daytime \| Rush hour | 0.173862 | 0.178708 | 0.174438 | 0.175753 | 
| 4 | K-means clusters \| Boroughs \| Years \| Months \| Seasons \| Hour of day \| Daytime \| Rush hour \| Day of week | 0.173933 |0.185982 | 0.174892 | 0.174815 |
| 5 | K-means clusters \| Boroughs \| Years \| Months \| Seasons \| Hour of day \| Daytime \| Rush hour \| Day of week \| Street names | 0.193007 | 0.181778 | 0.190340 | 0.176477 |
| 6 | K-means clusters \| Boroughs \| Years \| Months \| Seasons \| Hour of day \| Daytime \| Rush hour \| Day of week \| Street names \| Cross street names | 0.185801 | 0.228225 | 0.182672 | 0.182068 | 
