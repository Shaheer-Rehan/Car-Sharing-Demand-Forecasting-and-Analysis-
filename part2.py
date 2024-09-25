"""This script contains the code for the Data Analytics part of the project, which involves preprocessing the CarSharing dataset,
performing hypothesis testing to determine the relationship between each column and the demand rate,
checking for seasonal or cyclic patterns in the data, and building predictive models using ARIMA, Random Forest Regressor, and Deep Neural Network."""

import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pmdarima as pm
from sklearn.preprocessing import MinMaxScaler
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense

from part1 import df

"""Task 1: Import the CarSharing table into a CSV file and preprocess it with python. 
You need to drop duplicate rows and deal with null values using appropriate methods."""

df.drop_duplicates(inplace = True)

df['timestamp'] = pd.to_datetime(df['timestamp'], errors = 'coerce')

df_raw = df.copy()
df_raw.dropna(inplace = True)

df['temp'] = df['temp'].interpolate()
df['temp_feel'] = df['temp_feel'].interpolate()
df['humidity'] = df['humidity'].interpolate()
df['windspeed'] = df['windspeed'].interpolate()
df['demand'] = df['demand'].interpolate()

scaler = MinMaxScaler()

df_normalized = df.copy()

for column in ['temp', 'temp_feel', 'humidity', 'windspeed', 'demand']: # Check the correlation between the raw and preprocessed columns and normalize the columns
    correlation = df_raw[column].corr(df[column])
    print(f"The correlation between the raw and preprocessed {column} columns is {correlation:.2f}")
    df_normalized[column] = scaler.fit_transform(df_normalized[column].values.reshape(-1, 1))
    
df.to_csv('CarSharing_new.csv', index = False)
df_normalized.to_csv('CarSharing_normalized.csv', index = False)


"""Task 2: Using appropriate hypothesis testing, determine if there is a significant relationship 
between each column (except the timestamp column) and the demand rate. Report the tests’ results."""

# Check the relationship between season and demand
grouped_season = df_normalized.groupby('season')['demand'].apply(list) # Group the data by season and demand
f_statistic, p_value = stats.f_oneway(*grouped_season) # Perform the one-way ANOVA test
print(f"\nThe p-value for the one-way ANOVA test between season and demand is {p_value:.2f}. The F-statistic is {f_statistic:.2f}")

# Check the relationship between holiday and demand
grouped_holiday = df_normalized.groupby('holiday')['demand'].apply(list) # Group the data by holiday and demand
f_statistic, p_value = stats.f_oneway(*grouped_holiday) # Perform the one-way ANOVA test
print(f"\nThe p-value for the one-way ANOVA test between holiday and demand is {p_value:.2f}. The F-statistic is {f_statistic:.2f}")

# Check the relationship between workingday and demand
grouped_workingday = df_normalized.groupby('workingday')['demand'].apply(list) # Group the data by workingday and demand
f_statistic, p_value = stats.f_oneway(*grouped_workingday) # Perform the one-way ANOVA test
print(f"\nThe p-value for the one-way ANOVA test between workingday and demand is {p_value:.2f}. The F-statistic is {f_statistic:.2f}")

# Check the relationship between weather and demand
grouped_weather = df_normalized.groupby('weather')['demand'].apply(list) # Group the data by weather and demand
f_statistic, p_value = stats.f_oneway(*grouped_weather) # Perform the one-way ANOVA test
print(f"\nThe p-value for the one-way ANOVA test between weather and demand is {p_value:.2f}. The F-statistic is {f_statistic:.2f}")

# Check the relationship between temp and demand
y = df_normalized['demand']
X_temp = sm.add_constant(df_normalized['temp']) # Add a constant to the temp column
model_temp = sm.OLS(y, X_temp).fit() # Fit the OLS model
print(f"\nSummary of the OLS regression model between temp and demand: \n{model_temp.summary()}")

# Check the relationship between temp_feel and demand
X_tempfeel = sm.add_constant(df_normalized['temp_feel']) # Add a constant to the temp_feel column
model_tempfeel = sm.OLS(y, X_tempfeel).fit() # Fit the OLS model
print(f"\nSummary of the OLS regression model between temp_feel and demand: \n{model_tempfeel.summary()}")

# Check the relationship between humidity and demand
X_humidity = sm.add_constant(df_normalized['humidity']) # Add a constant to the humidity column
model_humidity = sm.OLS(y, X_humidity).fit() # Fit the OLS model
print(f"\nSummary of the OLS regression model between humidity and demand: \n{model_humidity.summary()}")

# Check the relationship between windspeed and demand
X_windspeed = sm.add_constant(df_normalized['windspeed']) # Add a constant to the windspeed column
model_windspeed = sm.OLS(y, X_windspeed).fit() # Fit the OLS model
print(f"\nSummary of the OLS regression model between windspeed and demand: \n{model_windspeed.summary()}")


"""Task 3: Please describe if you see any seasonal or cyclic pattern in the temp, humidity, windspeed, or demand data in 2017. 
Describe your answers."""

df_normalized_2017 = df_normalized[df_normalized['timestamp'].dt.year == 2017].copy() # Filter the data for the year 2017
df_normalized_2017.drop(columns = ['season', 'holiday', 'workingday', 'weather'], inplace = True) # Drop the non-numeric columns
df_normalized_2017.set_index('timestamp', inplace = True)
df_normalized_2017_daily = df_normalized_2017.resample('D').mean() # Resample the data to daily frequency
df_normalized_2017_daily.interpolate(method = 'time', inplace = True) # Interpolate the missing values

# Check for cyclic patterns in each column using the HP filter and plot the cyclical component
for column in ['temp', 'humidity', 'windspeed', 'demand']:
    cycle, trend = sm.tsa.filters.hpfilter(df_normalized_2017_daily[column], lamb = 100) # Apply the HP filter
    df_normalized_2017_daily.loc[:, f'{column}_cycle'] = cycle # Add the cyclical component to the dataframe
    df_normalized_2017_daily.loc[:, f'{column}_trend'] = trend # Add the trend component to the dataframe
    df_normalized_2017_daily[f'{column}_cycle'].plot() # Plot the cyclical component
    plt.title(f'Cyclical component of {column} in 2017')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()

# Generate the ACF plot for each column to get more idea about cyclic and seasonal patterns
for column in ['temp', 'humidity', 'windspeed', 'demand']:
    fig, ax = plt.subplots(figsize = (12, 6))
    plot_acf(df_normalized_2017_daily[column], lags = 50, ax = ax) # Generate the ACF plot
    plt.title(f'ACF plot of {column} in 2017')
    plt.tight_layout()
    plt.show()


# Check for seasonal patterns in each column using the OSCB test
for column in ['temp', 'humidity', 'windspeed', 'demand']:
    result = pm.arima.OCSBTest(12).estimate_seasonal_differencing_term(df_normalized_2017_daily[column]) # Perform the OSCB test. Setting argument to 30 for monthly seasonality
    print(f"\nThe seasonal differencing term for {column} in 2017 is {result}")


"""Task 4: Use an ARIMA model to predict the weekly average demand rate. Consider 30 percent of data for testing."""

df_normalized_numeric = df_normalized.drop(columns = ['season', 'holiday', 'workingday', 'weather']) # Drop the non-numeric columns
df_normalized_numeric.set_index('timestamp', inplace = True)
df_normalized_weekly = df_normalized_numeric.resample('W').mean() # Resample the data to weekly frequency
df_normalized_weekly.interpolate(method = 'time', inplace = True) # Interpolate the missing values

# Split the data into training and testing sets
train_size = int(0.7 * len(df_normalized_weekly))
train, test = df_normalized_weekly[:train_size], df_normalized_weekly[train_size:]

# Fit the ARIMA model
arima_model = pm.auto_arima(train['demand'], seasonal = True, m = 12, stepwise = True, suppress_warnings = True, error_action = 'ignore', trace = True)
print(f"\nSummary of the ARIMA model: \n{arima_model.summary()}")
arima_model.plot_diagnostics(figsize = (12, 6))
plt.tight_layout()
plt.title('Diagnostics of the ARIMA model')
plt.show()

# Make predictions using the ARIMA model
arima_predictions = arima_model.predict(n_periods = len(test))
arima_predictions = pd.Series(arima_predictions, index = test.index)

# Plot the actual and predicted values
plt.figure(figsize = (12, 6))
plt.plot(train['demand'], label = 'Training data')
plt.plot(test['demand'], label = 'Actual demand')
plt.plot(arima_predictions, label = 'Predicted demand')
plt.title('Actual vs Predicted demand using ARIMA model')
plt.xlabel('Time')
plt.ylabel('Demand')
plt.legend()
plt.tight_layout()
plt.show()

# Calculate the RMSE
rmse_arima = np.sqrt(np.mean((arima_predictions - test['demand'])**2))
print(f"\nThe RMSE of the ARIMA model is {rmse_arima:.2f}")


"""Task 5: Use a random forest regressor and a deep neural network to predict the demand rate and 
report the minimum square error for each model. Which one is working better? Why? Please describe the reason."""

# Random Forest Regressor
# Split the data into training and testing sets
X_train = train.drop(columns = 'demand')
y_train = train['demand']
X_test = test.drop(columns = 'demand')
y_test = test['demand']

# Fit the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators = 100, random_state = 1)
rf_model.fit(X_train, y_train)

# Make predictions using the Random Forest Regressor
rf_predictions = rf_model.predict(X_test)

# Calculate the RMSE
rmse_rf = np.sqrt(np.mean((rf_predictions - y_test)**2))
print(f"\nThe RMSE of the Random Forest Regressor is {rmse_rf:.2f}")

# Plot the actual vs predicted values
plt.figure(figsize = (12, 6))
plt.scatter(y_test.index, y_test, label = 'Actual demand')
plt.plot(y_test.index, rf_predictions, label = 'Predicted demand')
plt.title('Actual vs Predicted demand using Random Forest Regressor')
plt.xlabel('Time')
plt.ylabel('Demand')
plt.legend()
plt.tight_layout()
plt.show()


# Deep Neural Network
# Split the data into training and testing sets
X_train = train.drop(columns = 'demand')
y_train = train['demand']
X_test = test.drop(columns = 'demand')
y_test = test['demand']

# Define the neural network model
NN_model = Sequential()
NN_model.add(Dense(128, input_dim = X_train.shape[1], activation = 'relu'))
NN_model.add(Dense(64, activation = 'relu'))
NN_model.add(Dense(1, activation = 'linear'))
NN_model.compile(loss = 'mean_squared_error', optimizer = 'adam')

# Fit the neural network model
NN_model.fit(X_train, y_train, epochs = 100, batch_size = 32, validation_split = 0.2, verbose = 0)

# Make predictions using the neural network model
NN_predictions = NN_model.predict(X_test).flatten()

# Calculate the RMSE
rmse_NN = np.sqrt(np.mean((NN_predictions - y_test)**2))
print(f"\nThe RMSE of the Neural Network model is {rmse_NN:.2f}")

# Plot the actual vs predicted values
plt.figure(figsize = (12, 6))
plt.scatter(y_test.index, y_test, label = 'Actual demand')
plt.plot(y_test.index, NN_predictions, label = 'Predicted demand')
plt.title('Actual vs Predicted demand using Neural Network model')
plt.xlabel('Time')
plt.ylabel('Demand')
plt.legend()
plt.tight_layout()
plt.show()

# The Random Forest Regressor has a lower RMSE compared to the Neural Network model. This could be due to the fact that the Random Forest Regressor 
# is able to capture non-linear relationships and interactions between features more effectively than the Neural Network model as the Random Forest
# model is an ensemble of decision trees. The Neural Network model may require more tuning of hyperparameters and architecture to improve its performance.