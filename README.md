# NYC MVA model

Between 2012 and 2020, there were close to 2 million motor vehicle accidents in New York City. A significant number of these accidents involved non-drivers, such as pedestrians, cyclists, etc. This model attempts to predict accidents involving pedestrians, with the purpose of determining patterns for new traffic recommendations and police presence.

The data was collected a number of months ago from the [NYC OpenData](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95) dataset. The dataset is constantly updated, and as such, the current data contains more accidents than the models trained here. 

## Instructions
All tools are stored in the _Predictor tools_ folder. The _Predictor_ program will prompt the user for relevant data and will present a prediction based on the features described below. The _Retrain model_ program will allow the user to update the model based on the most current data available on NYC OpenData. Please note that this will take several hours, owing to an API that requires single-item requests. I will look into multithreading (perhaps `ThreadPoolExecutor`) as an option for improving speed. 

## Model details
### Target variable
The data contains geographical and borough information, along with date and time of each accident. A contributing factor was applied to each accident, as well as the vehicle types. The data set also contains the number of persons injured or killed; these are then broken down into pedestrians, cyclists, and motorists injured or killed. To develop a target, I combined the `NUMBER OF PEDESTRIANS INJURED`, `NUMBER OF PEDESTRIANS KILLED`, `NUMBER OF CYCLISTS INJURED`, and `NUMBER OF CYCLISTS KILLED`  into a single column called `TOTAL PEDESTRIAN CASUALTIES`. I then created the target based on the nonzero records contained in the `TOTAL PEDESTRIAN CASUALTIES` column, simply called `CASUALTIES?` and containing a 1 for all nonzero records, and a 0 for all others.

### Model development
See more detail on the features in the _EDA_ file in the _Model prework_ folder.

#### Classifiers/Metrics
In such a model, the ideal metric is recall, owing to the imbalanced nature of the data and the desire to minimize false negatives--it is better to err on the side of caution than to assume that an accident will _not_ take place.
Each feature set was examined with a Logistic Regression and Random Forest Classifier. Bayesian hyperparameter tuning was conducted on the Random Forest Classifier using Feature Set 1 below and the `scikit-optimize` [library](https://scikit-optimize.github.io/stable/); see the _Hyperparamter tuning_ notebook in the _Model prework_ folder for details. The following table presents which hyperparameters were tuned:

| Hyperparameter | Levels |
| :--- | :---: |
| **Criterion** | Gini, entropy |
| **Max depth** | Integers 0-50 |
| **Min samples leaf** | Integers 1-10 |
| **N estimators** | Integers 1-100 |
| **Max features** | "None" and integers 1-10 |
| **Min impurity decrease** | 0, 0.05, 0.1, 0.15, 0.2, 0.25 |

To conduct the hyperparameter tuning, an F1 score was used in order to penalize false positives as well, thus preventing a false optimization of hyperparameters through maximization of recall at the expense of precision.

#### Feature set 1: Clusters/Boroughs
The first data examined was the geographical data. There was a significant amount of data that was either mislabeled or incomplete. I removed all accident data with incomplete or otherwise incorrect latitude and longitude data, i.e. all records whose latitude and longitude data placed them outside of the bounds of NYC. I then corrected all ZIP code data based on the latitudes and longitudes. This data served as the first iteration of model development, Because each borough has different rates of accident and different "culture of driving," so to speak, I ran a K-means cluster test using 2-30 clusters on each borough and implemented the cluster count with the highest F1 score for each borough:

![K-means hyperparameters](Model%20prework/K-Means%20borough%20analysis.png)

See _Hyperparameter tuning.py_ for details. As with the Random Forest Classifier hyperparameter tuning, I decided to implement an F1 metric in order to avoid maximizing recall at the expense of precision. The cluster counts are saved in the _Predictor tools_ folder as a pickled dictionary. These cluster counts, combined with a dummy set of the borough feature, defined the first model.

#### Feature set 2: Year/Month/Season
While a specific _pattern_ is not easily detectable among the date-related data, it is apparent that these are nonetheless relevant features:

![Seasons by hour](Image%20resources/Seasons.png)

![Years](Image%20resources/Years.png)

Intuition dictates examination of such--for example, it is highly likely that in 2020 there are fewer cases of pedestrian-related accidents, owing to reduced foot traffic caused by COVID restrictions. 2014-2016 are unintuitively low, however. Similarly, it stands to reason that seasons may be a relevant feature--camps, not schools, are open in the summer, reducing bus traffic and children in the streets, and winter is well-known as the tourist season in NYC. However, winter has an unusual spike at 6 PM, and fall has an unusual spike at 5 PM. Spring is unusually low across the board. A more granular examination is the month, but trends should follow similar patterns based on rationale.

#### Feature set 3: Hour/Daytime/Rush hour
Depending on the day, more accidents occur during certain hours than others:

![Weekday by hour](Image%20resources/Weekdays.png)

The raw hour feature is an obvious consideration; additional time features included are daytime--each day is calculated using the `astral` module--and whether or not the accident took place during rush hour, calculated as between 5-10 AM and 4-8 PM.

#### Feature set 4: Weekdays
As demonstrated in the previous feature set, certain days have very different casualty patterns than others. The fourth feature set is therefore the days of the week.

#### Feature set 5: Street names
Understandably, the street name is of particular relevance when predicting whether an accident will occur involving a pedestrian. EDA demonstrated that the Euclidean distance of the two farthest accidents of a street–i.e., the streets whose accidents cover the largest area–correlates with number of pedestrian casualties with a Pearson’s correlation coefficient of 0.86. Curiously, however, the number of accidents only correlates at a coefficient of 0.62. Future improvements may include a multiplier to each record based on the rate of casualties to Euclidean range, thereby weighting the more “dangerous” streets more heavily.

Regardless, I stemmed the data to clean common but different street titles–“Ave” versus “Avenue,” for example. I then transformed the via a count vectorizer. This provided weight for street types, such as Roads versus Avenues, while providing a full dummy feature set for all street names.

#### Feature set 6: Cross street names
While the street is of vital importance when predicting accident occurrence, of similar importance is the cross street. A similar process to the cleaning and vectorizing of Feature Set 5 was performed on this data.

#### Feature set 7: Contributing factors
Only certain types of accidents are preventable with police presence. For example, an accident caused by vehicle malfunction is not preventable. Because there is no way to prevent such an accident, we have no need to try to predict such an accident.

## Feature comparison
The following graph presents the recall scores for each model, based on Logistic Regression and the Random Forest Classifier.


## Future improvements
Weather plays an important role in driving. I would like to join a weather dataset to the current data with generalized weather features such as can be input by a user (e.g. “sunny,” “cloudy,” “raining,” “snowing,” “foggy,” etc.) I would like to conduct mitigation operations on the data to correct for the imbalance, to try to improve the score. Because there are many highways that both muddle the clustering and damage the predictive quality of the data set (because there are few, if any, pedestrian casualties on highways), I would like to remove the highway accidents from the data during the cleaning process. This is not a straightforward process; there are highways, such as the FDR Drive or the Harlem River Drive, whose name shares commonalities with local roads; there are local roads, such as Kings Highway or Eastern Parkway, whose names share commonalities with highways. I would like to compile a list of NYC highway names and remove data containing those names. 
