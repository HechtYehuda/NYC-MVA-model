# Summary of models

The _Models_ folder contains the Python code to run each model. Models are numbered accordingly. Data to be added to a DropBox and linked shortly.

Recall scores were used to calculate model accuracy, owing to the imbalanced nature of the data and the desire to minimize false negatives. LR = Logistic Regression; RF = Random Forest.

| Model # | Predictors | LR Train | RF Train | LR Test | RF Test |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | K-means clusters <br/> Boroughs | 0.64391 | 0.63672 | 0.64285 | 0.63650 |
| 2 | K-means clusters <br/> Boroughs <br/> Years <br/> Months <br/> Seasons | 0.63104 | 0.63088 | 0.62935 | 0.53634 |
| 3 | K-means clusters <br/> Boroughs <br/> Years <br/> Months <br/> Seasons <br/> Hour of day <br/> Daytime <br/> Rush hour | 0.61001 | 0.61142 | 0.61115 | 0.58250 | 
| 4 | K-means clusters <br/> Boroughs <br/> Years <br/> Months <br/> Seasons <br/> Hour of day <br/> Daytime <br/> Rush hour <br/> Day of week | 0.60957 |0.62702 | 0.61217 | 0.49460 |
| 5 | K-means clusters <br/> Boroughs <br/> Years <br/> Months <br/> Seasons <br/> Hour of day <br/> Daytime <br/> Rush hour <br/> Day of week <br/> Street names | 0.67881 | 0.69004 | 0.66485 | 0.63160 |
| 6 | K-means clusters <br/> Boroughs <br/> Years <br/> Months <br/> Seasons <br/> Hour of day <br/> Daytime <br/> Rush hour <br/> Day of week <br/> Street names <br/> Cross street names | 0.639238 | 0.63035 | 0.62541 | 0.58482 | 
| 7 | K-means clusters <br/> Boroughs <br/> Years <br/> Months <br/> Seasons <br/> Hour of day <br/> Daytime <br/> Rush hour <br/> Day of week <br/> Street names <br/> Cross street names <br/> Contributing factors | 0.68684 | 0.62363 | 0.67290 | 0.62363 |