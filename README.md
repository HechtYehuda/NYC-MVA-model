# NYC MVA model

Between 2012 and 2020, there were close to 2 million motor vehicle accidents in New York City. A significant number of these accidents involved non-drivers, such as pedestrians, cyclists, etc. This model attempts to predict accidents involving pedestrians, with the purpose of determining patterns for new traffic recommendations and police presence.

The data was collected a number of months ago from the [NYC OpenData](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95) dataset. The dataset is constantly updated, and as such, the current data contains more accidents than the models trained here. 

## Model details
### Target variable
The data contains geographical and borough information, along with date and time of each accident. A contributing factor was applied to each accident, as well as the vehicle types. The data set also contains the number of persons injured or killed; these are then broken down into pedestrians, cyclists, and motorists injured or killed. To develop a target, I combined the `NUMBER OF PEDESTRIANS INJURED`, `NUMBER OF PEDESTRIANS KILLED`, `NUMBER OF CYCLISTS INJURED`, and `NUMBER OF CYCLISTS KILLED`  into a single column called `TOTAL PEDESTRIAN CASUALTIES`. I then created the target based on the nonzero records contained in the `TOTAL PEDESTRIAN CASUALTIES` column, simply called `CASUALTIES?` and containing a 1 for all nonzero records, and a 0 for all others.

### Model development

#### Classifiers/Metrics
In such a model, the ideal metric is recall, owing to the imbalanced nature of the data and the desire to minimize false negatives--it is better to err on the side of caution than to assume that an accident will _not_ take place.

#### Model 1: Clusters/Boroughs
The first data examined was the geographical data. There was a significant amount of data that was either mislabeled or incomplete. I removed all accident data with incomplete or otherwise incorrect latitude and longitude data, i.e. all records whose latitude and longitude data placed them outside of the bounds of NYC. I then corrected all ZIP code data based on the latitudes and longitudes. This data served as the first iteration of model development, Because each borough has different rates of accident and different "culture of driving," so to speak, I ran a K-means cluster test using 2-20 on each borough and implemented the cluster count with the highest recall score for each borough. See _Hyperparameter tuning.py_ and the accompanying _K-Means tuning_ PNG file for details. These cluster counts, combined with a dummy set of the borough feature, defined the first model.

#### Model 2: Year/Month/Season
While a specific _pattern_ is not easily detectable among the date-related data, it is apparent that these are nonetheless relevant features. Intuition dictates examination of such--for example, it is highly likely that in 2020 there are fewer cases of pedestrian-related accidents, owing to reduced foot traffic caused by COVID restrictions. Similarly, it stands to reason that seasons may be a relevant features, 