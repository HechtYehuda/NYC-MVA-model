# Summary of models

The _Models_ folder contains the Python code to run each model. Models are numbered accordingly. Data to be added to a DropBox and linked shortly.

Recall scores were used to calculate model accuracy, owing to the imbalanced nature of the data and the desire to minimize false negatives. LR = Logistic Regression; RF = Random Forest.

| Model # | Predictors | LR Train | RF Train | LR Test | RF Test |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | K-means clusters <br/> Boroughs | 0.61629 | 0.64351 | 0.61824 | 0.64517 |
| 2 | K-means clusters <br/> Boroughs <br/> Years <br/> Months <br/> Seasons | 0.62645 | 0.61663 | 0.62003 | 0.53417 |
| 3 | K-means clusters <br/> Boroughs <br/> Years <br/> Months <br/> Seasons <br/> Hour of day <br/> Daytime <br/> Rush hour | 0.60990 | 0.65685 | 0.60519 | 0.40945 | 
| 4 | K-means clusters <br/> Boroughs <br/> Years <br/> Months <br/> Seasons <br/> Hour of day <br/> Daytime <br/> Rush hour <br/> Day of week | 0.60974 |0.65514 | 0.60449 | 0.38259 |
| 5 | K-means clusters <br/> Boroughs <br/> Years <br/> Months <br/> Seasons <br/> Hour of day <br/> Daytime <br/> Rush hour <br/> Day of week <br/> Street names | 0.67963 | 0.71601 | 0.66516 | 0.67795 |
| 6 | K-means clusters <br/> Boroughs <br/> Years <br/> Months <br/> Seasons <br/> Hour of day <br/> Daytime <br/> Rush hour <br/> Day of week <br/> Street names <br/> Cross street names | 0.64065 | 0.59087 | 0.62070 | 0.55425 | 
| 7 | K-means clusters <br/> Boroughs <br/> Years <br/> Months <br/> Seasons <br/> Hour of day <br/> Daytime <br/> Rush hour <br/> Day of week <br/> Street names <br/> Cross street names <br/> Contributing factors | 0.68923 | 0.58489 | 0.67461 | 0.56477 |
