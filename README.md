# Predicting Bike Sharing Patterns
- Prediction of bike rental count hourly or daily based on the environmental and seasonal settings using neural networks via Pytorch.
- type of the problem: __Regression problem__


## Background

Bike sharing systems are new generation of traditional bike rentals where whole process from membership, rental and return back has become automatic. Through these systems, user is able to easily rent a bike from a particular position and return  back at another position.

Currently, there are about over 500 bike-sharing programs around the world which is composed of over 500 thousands bicycles. Today, there exists great interest in these systems due to their important role in traffic,  environmental and health issues. Apart from interesting real world applications of bike sharing systems, the characteristics of data being generated by these systems make them attractive for the research.

Opposed to other transport services such as bus or subway, the duration of travel, departure and arrival position is explicitly recorded in these systems. This feature turns bike sharing system into a virtual sensor network that can be used for sensing mobility in the city.

Hence, it is expected that most of important events in the city could be detected via monitoring these data.



## Dataset

Bike-sharing rental process is highly correlated to the environmental and seasonal settings. For instance, weather conditions, precipitation, day of week, season, hour of the day, etc. can affect the rental behaviors.

The core data set is related to the two-year historical log corresponding to years 2011 and 2012 from Capital Bikeshare system, Washington D.C., USA which is publicly available in the [Capital Bikeshare system data][capitalbikeshare-url].



### Files
  - [bike-sharing-patterns.csv](dataset/raw/bike-sharing-patterns.csv): bike sharing counts aggregated on hourly basis. Records: 17379 hours



### raw-dataset-sample
![raw-dataset-sample.png](raw-dataset-sample.png)

### Dataset characteristics
  - `instant`: record index
  - `dteday`: date
  - `season`: season (1:springer, 2:summer, 3:fall, 4:winter)
  - `yr`: year (0: 2011, 1:2012)
  - `mnth`: month (1 to 12)
  - `hr`: hour (0 to 23)
  - `holiday`: weather [day is holiday or not][holiday-schedule-url]
  - `weekday`: day of the week
  - `workingday`: if day is neither weekend nor holiday is 1, otherwise is 0.
  - `weathersit`:
    1. Clear, Few clouds, Partly cloudy, Partly cloudy
    2. Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
    3. Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
    4. Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
  - `temp`: Normalized temperature in Celsius. The values are divided to 41 (max)
  - `atemp`: Normalized feeling temperature in Celsius. The values are divided to 50 (max)
  - `hum`: Normalized humidity. The values are divided to 100 (max)
  - `windspeed`: Normalized wind speed. The values are divided to 67 (max)
  - `casual`: count of casual users
  - `registered`: count of registered users
  - `cnt`: count of total rental bikes including both casual and registered

___

## Model architecture
