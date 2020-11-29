# NYC MVA model

Between 2012 and 2020, there were close to 2 million motor vehicle accidents in New York City. A significant number of these accidents involved non-drivers, such as pedestrians, cyclists, etc. This model attempts to predict accidents involving pedestrians, with the purpose of determining patterns for new traffic recommendations and police presence.

The data was collected a number of months ago from the [NYC OpenData](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95) dataset. The dataset is constantly updated, and as such, the current data contains more accidents than the models trained here. 

The data contains geographical and borough information, along with date and time of each accident. A contributing factor was applied to each accident, as well as the vehicle types. The data set also contains the number of persons injured or killed; these are then broken down into pedestrians, cyclists, and motorists injured or killed. To develop a target, I combined` the `NUMBER OF PEDESTRIANS INJURED`, `NUMBER OF PEDESTRIANS KILLED`, `NUMBER OF CYCLISTS INJURED`, and `NUMBER OF CYCLISTS KILLED`  into a single column called `TOTAL PEDESTRIAN CASUALTIES`. I then created the target based on the nonzero records contained in the `TOTAL PEDESTRIAN CASUALTIES` column, simply called `CASUALTIES?` and containing a 1 for all nonzero records, and a 0 for all others.

### Model development
The first data examined was the geographical data. There was a significant amount of data that was either mislabeled or incomplete. I removed all accident data with incomplete or otherwise incorrect latitude and longitude data, i.e. all records whose latitude and longitude data placed them outside of the bounds of NYC. I then corrected all ZIP code data based on the latitudes and longitudes. This data served as the first iteration of model development, I ran a K-means cluster test using 2-100 clusters and a random forest classifier with an F1 score function; the best score came from 52 clusters. These 52 clusters, along with the borough data, served as the feature set for the first model.
 
The current model has been developed based on an examination of the data. I noticed patterns in the date/time data suggesting that factors such as hour of the day and day of the week may be relevant factors for determining the accidents.
