# NYC-MVA-model

A model to predict pedestrian casualties based on accidents in NYC between 2012 and 2020.

The _Models_ folder contains the Python code to run each model. Data to be added to a DropBox and linked shortly. 

| Model # | Predictors | LR Train | RF Train | LR Test | RF Test |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | K-means clusters \| Boroughs | 0.166508 | 0.166507 | 0.166727 | 0.168079 |
| 2 | K-means clusters \| Boroughs \| Years \| Months \| Seasons ||| 0.167534 | 0.168375 |
| 3 | K-means clusters \| Boroughs \| Years \| Months \| Seasons \| Hour of day \| Daytime? \| Rush hour? ||| 0.174438 | 0.175753 | 
| 4 | K-means clusters \| Boroughs \| Years \| Months \| Seasons \| Hour of day \| Daytime? \| Rush hour? \| Day of week ||| 0.174892 | 0.174815 |