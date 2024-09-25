# Car Sharing Demand Forecasting & Analysis
This repository contains the python script files for a car-sharing demand prediction project using a dataset spanning January 2017 to August 2018. The project is divided into two main parts:

- **Part 1: Database management (SQLite)** – involves using SQLite via python for table creation, column modifications, querying, and calculating demand rates.
- **Part 2: Data analytics** – includes preprocessing, hypothesis testing, time series analysis, and predictive modeling using ARIMA, Random Forest Regressor, and Deep Neural Network (DNN) models.


## Data
The dataset includes hourly records for weather, temperature, humidity, wind speed, and demand from January 2017 to August 2018. The dataset can be accessed through the CarSharing.csv file in this repository.

## Requirements
To run the code, the required Python libraries can be installed as follows:
pip install pandas numpy scipy statsmodels pmdarima matplotlib scikit-learn keras

## Part 1: Database Management
The first part of the project involves working with SQLite to create a relational database for the car-sharing dataset. The key steps of the database management performed are:
- Creating tables for storing the car-sharing data.
- Performing operations like adding, modifying, and querying columns.
- Calculating and storing the demand rates based on specific conditions.  

Refer to part1.py for the detailed implementation of database management tasks.

## Part 2: Data Analytics
Refer to part2.py for the detailed implementation of data analytics tasks.

### Preprocessing
The main steps performed during preprocessing were as follows:
1. Dropped duplicate rows.
2. Handled missing values using interpolation.
3. Normalized key features using MinMaxScaler.

### Hypothesis Testing
Used hypothesis testing (ANOVA and OLS regression) to determine relationships between various features (like season, holiday, working day, weather, temperature, humidity, windspeed) and the demand rate.

### Seasonal/Cyclic Pattern Analysis
Analyzed cyclic patterns and seasonality in key features such as temperature, humidity, windspeed, and demand using:
- HP filters for extracting trend and cyclical components.
- Autocorrelation plots (ACF) for visualizing patterns.
- OCSB test to check for seasonal differencing terms.

### Predictive Models
**1. ARIMA Model**  
- Used ARIMA to forecast the weekly average demand rate.
- Split the data (70% training and 30% testing) and evaluated the model's performance using RMSE.
- Visualized the predicted demand vs. actual demand.

**2. Random Forest Regressor**  
- Built a Random Forest Regressor model to predict demand.
- Plotted the predicted vs. actual demand.

**3. Deep Neural Network**  
- Built a Deep Neural Network using Keras to predict demand.
- Plotted the predicted vs. actual demand.
