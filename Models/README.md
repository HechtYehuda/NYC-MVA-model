# Summary of models

The _Models_ folder contains the Python code to run each model. Models are numbered accordingly. Data to be added to a DropBox and linked shortly.

Recall scores were used to calculate model accuracy, owing to the imbalanced nature of the data and the desire to minimize false negatives. LR = Logistic Regression; RF = Random Forest.

| Model # | Predictors | LR Train | RF Train | LR Test | RF Test |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | K-means clusters <br/> Boroughs | 0.707050 | 0.602465 | 0.707783 | 0.601080 |
| 2 | K-means clusters <br/> Boroughs <br/> Years <br/> Months <br/> Seasons | 0.637117 | 0.611720 | 0.639231 | 0.602192 |
| 3 | K-means clusters <br/> Boroughs <br/> Years <br/> Months <br/> Seasons <br/> Hour of day <br/> Daytime <br/> Rush hour | 0.173862 | 0.178708 | 0.174438 | 0.175753 | 
| 4 | K-means clusters <br/> Boroughs <br/> Years <br/> Months <br/> Seasons <br/> Hour of day <br/> Daytime <br/> Rush hour <br/> Day of week | 0.173933 |0.185982 | 0.174892 | 0.174815 |
| 5 | K-means clusters <br/> Boroughs <br/> Years <br/> Months <br/> Seasons <br/> Hour of day <br/> Daytime <br/> Rush hour <br/> Day of week <br/> Street names | 0.196917 | 0.193533 | 0.193701 | 0.185183 |
| 6 | K-means clusters <br/> Boroughs <br/> Years <br/> Months <br/> Seasons <br/> Hour of day <br/> Daytime <br/> Rush hour <br/> Day of week <br/> Street names <br/> Cross street names | 0.189979 | 0.228807 | 0.186668 | 0.186783 | 
| 7 | K-means clusters <br/> Boroughs <br/> Years <br/> Months <br/> Seasons <br/> Hour of day <br/> Daytime <br/> Rush hour <br/> Day of week <br/> Street names <br/> Cross street names <br/> Contributing factors | 0.226696 | 0.224418 | 0.222967 | 0.211143 |
