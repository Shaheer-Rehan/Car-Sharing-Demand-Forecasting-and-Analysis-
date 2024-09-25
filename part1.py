"""This script performs the following tasks:
1. Creates an SQLite database and imports the data from the 'CarSharing.csv' file into a table named 'CarSharing'.
2. Adds a column to the 'CarSharing' table named 'temp_category' containing three distinct string values based on the 'temp_feel' column.
3. Creates another table named 'temperature' by selecting the 'temp', 'temp_feel', and 'temp_category' columns from the 'CarSharing' table.
4. Assigns a number to each distinct string value in the 'weather' column and adds another column named 'weather_code' to the 'CarSharing' table.
5. Creates a table called 'weather' and copies the columns 'weather' and 'weather_code' to this table.
6. Creates a table called 'time' containing the timestamp, hour, weekday name, and month name of each row.
7.1. Finds the date and time with the highest demand rate in 2017.
7.2. Creates a table containing the name of the weekday, month, and season with the highest and lowest average demand rates throughout 2017.
7.3. Creates a table showing the average demand rate at different hours of a selected weekday throughout 2017.
7.4. Determines the weather conditions in 2017 and calculates the average, highest, and lowest wind speed and humidity for each month in 2017.
     Also, creates a table showing the average demand rate for each cold, mild, and hot weather in 2017.
7.5. Creates a table showing the information requested in 7.4 for the month with the highest average demand rate in 2017.
The script uses the sqlite3 library to interact with the SQLite database and the pandas library 
to read the data from the 'CarSharing.csv' file and write it to the database.
The tasks are performed using SQL queries executed on the database."""


import pandas as pd
import sqlite3 as sql
from datetime import datetime
import matplotlib.pyplot as plt


"""Task 1: Create an SQLite database and import the data into a table named “CarSharing”. 
Create a backup table and copy the whole table into it."""

CarSharing = pd.read_csv('CarSharing.csv')
conn = sql.connect('CarSharing.db', isolation_level = None)

CarSharing.to_sql('CarSharing', conn, if_exists = 'replace', index = False) # Write the data to the database table 'CarSharing' in the database 'CarSharing.db'
conn.executescript("""
                    BEGIN;
                   
                    DROP TABLE IF EXISTS CarSharing_backup;
                    CREATE TABLE CarSharing_backup AS SELECT * FROM CarSharing;
                    
                    COMMIT;
                    """)


"""Task 2: Add a column to the CarSharing table named “temp_category”. 
This column should contain three string values. If the “feels-like” temperature is less than 10 
then the corresponding value in the temp_category column should be “Cold”, 
if the feels-like temperature is between 10 and 25, the value should be “Mild”, 
and if the feels-like temperature is greater than 25, then the value should be “Hot”."""

conn.executescript("""
                    BEGIN;
                   
                    ALTER TABLE CarSharing ADD COLUMN temp_category TEXT;
                    UPDATE CarSharing SET temp_category = "Cold" WHERE "temp_feel" < 10;
                    UPDATE CarSharing SET temp_category = "Mild" WHERE "temp_feel" BETWEEN 10 AND 25;
                    UPDATE CarSharing SET temp_category = "Hot" WHERE "temp_feel" > 25;
                   
                    COMMIT;
                    """) 


"""Task 3: Create another table named “temperature” by selecting the temp, temp_feel, and temp_category columns. 
Then drop the temp and temp_feel columns from the CarSharing table."""

conn.executescript("""
                    BEGIN;
                   
                    DROP TABLE IF EXISTS Temperature;
                    CREATE TABLE Temperature AS SELECT id, temp, temp_feel, temp_category FROM CarSharing; -- Adding the id column to act as a foreign key
                   
                    DROP TABLE IF EXISTS CarSharing_new;
                    CREATE TABLE CarSharing_new AS SELECT id, timestamp, season, holiday, workingday, weather, humidity, windspeed, demand FROM CarSharing;
                   
                    DROP TABLE CarSharing;
                    ALTER TABLE CarSharing_new RENAME TO CarSharing;
                   
                    COMMIT;
                    """)


"""Task 4: Find the distinct values of the weather column and assign a number to each value. 
Add another column named “weather_code” to the table containing each row's assigned weather code."""

conn.executescript("""
                    BEGIN;
                   
                    SELECT DISTINCT weather FROM CarSharing;
                    ALTER TABLE CarSharing ADD COLUMN weather_code INTEGER;
                    UPDATE CarSharing SET weather_code = 1 WHERE weather = "Clear or partly cloudy";
                    UPDATE CarSharing SET weather_code = 2 WHERE weather = "Mist";
                    UPDATE CarSharing SET weather_code = 3 WHERE weather = "Light snow or rain";
                    UPDATE CarSharing SET weather_code = 4 WHERE weather = "heavy rain/ice pellets/snow + fog";
                   
                    COMMIT;
                    """)


"""Task 5: Create a table called “weather” and copy the columns “weather” and “weather_code” to this table. 
Then drop the weather column from the CarSharing table."""

conn.executescript("""
                    BEGIN;
                   
                    DROP TABLE IF EXISTS Weather;
                    CREATE TABLE Weather AS SELECT DISTINCT weather, weather_code FROM CarSharing;
                   
                    DROP TABLE IF EXISTS CarSharing_new1;
                    CREATE TABLE CarSharing_new1 AS SELECT id, timestamp, season, holiday, workingday, humidity, windspeed, demand, weather_code FROM CarSharing;
                   
                    DROP TABLE CarSharing;
                    ALTER TABLE CarSharing_new1 RENAME TO CarSharing;
                   
                    COMMIT;
                    """)
                    

"""Task 6: Create a table called time with four columns containing each row's timestamp, hour, weekday name, and month name
(Hint: you can use the strftime() function for this purpose)."""

conn.executescript("""
                    BEGIN;
                   
                    DROP TABLE IF EXISTS Time;
                    CREATE TABLE Time AS SELECT timestamp, 
                    strftime('%H', timestamp) AS hour, 
                    CASE strftime('%w', timestamp)
                        WHEN '0' THEN 'Sunday'
                        WHEN '1' THEN 'Monday'
                        WHEN '2' THEN 'Tuesday'
                        WHEN '3' THEN 'Wednesday'
                        WHEN '4' THEN 'Thursday'
                        WHEN '5' THEN 'Friday'
                        WHEN '6' THEN 'Saturday' 
                    END AS weekday, 
                    CASE strftime('%m', timestamp)
                        WHEN '01' THEN '01. January'
                        WHEN '02' THEN '02. February'
                        WHEN '03' THEN '03. March'
                        WHEN '04' THEN '04. April'
                        WHEN '05' THEN '05. May'
                        WHEN '06' THEN '06. June'
                        WHEN '07' THEN '07. July'
                        WHEN '08' THEN '08. August'
                        WHEN '09' THEN '09. September'
                        WHEN '10' THEN '10. October'
                        WHEN '11' THEN '11. November'
                        WHEN '12' THEN '12. December'
                    END AS month 
                    FROM CarSharing;
                   
                    COMMIT;
                    """) 
# The strftime() function is used to extract the hour, weekday, and month from the timestamp column. It cannot directly give weekday and month names.
# So, a CASE statement is used to map the numeric values to the corresponding names. This can also be done by converting the CarSharing table to a
# pandas dataframe and using the strftime() function to extract the weekday and month names and converting back to an SQLite table. While this would
# be easier and more readable, it is inefficient and computationally expensive.


"""Task 7.1: Please tell me which date and time we had the highest demand rate in 2017."""

highest_demand_2017 = conn.execute("""       
                                    SELECT timestamp, demand FROM CarSharing 
                                    WHERE demand = (SELECT MAX(demand) FROM CarSharing WHERE strftime('%Y', timestamp) = '2017') AND strftime('%Y', timestamp) = '2017';
                                    """).fetchall()
timestamp_highest_demand = datetime.strptime(highest_demand_2017[0][0], '%Y-%m-%d %H:%M:%S')
print("\nThe highest demand rate in 2017 was " + str(round(highest_demand_2017[0][1], 2)) + ". It occurred on " + 
      timestamp_highest_demand.strftime('%Y-%m-%d') + " at " + timestamp_highest_demand.strftime('%H:%M:%S') + ".\n")   


"""Task 7.2: Give me a table containing the name of the weekday, month, and season in which we had the highest and lowest average demand rates throughout 2017. 
Please include the calculated average demand values as well."""

conn.executescript("""
                    BEGIN;
                   
                    DROP TABLE IF EXISTS HighAvg;
                    CREATE TABLE HighAvg AS
                    SELECT X.weekday, X.month, X.season, MAX(X.avg_demand) AS avg_demand, 'Highest Avg Demand' AS demand_type FROM 
                    (SELECT T.weekday, T.month, C.season, AVG(C.demand) AS avg_demand
                    FROM CarSharing AS C
                    INNER JOIN Time AS T ON C.timestamp = T.timestamp                                             
                    WHERE strftime('%Y', C.timestamp) = '2017'
                    GROUP BY T.weekday, T.month, C.season) AS X;
                   
                    DROP TABLE IF EXISTS LowAvg;
                    CREATE TABLE LowAvg AS
                    SELECT Y.weekday, Y.month, Y.season, MIN(Y.avg_demand) AS avg_demand, 'Lowest Avg Demand' AS demand_type FROM 
                    (SELECT T.weekday, T.month, C.season, AVG(C.demand) AS avg_demand
                    FROM CarSharing AS C
                    INNER JOIN Time AS T ON C.timestamp = T.timestamp                                             
                    WHERE strftime('%Y', C.timestamp) = '2017'
                    GROUP BY T.weekday, T.month, C.season) AS Y;
                   
                    DROP TABLE IF EXISTS AvgDemand2017;
                    CREATE TABLE AvgDemand2017 AS
                    SELECT * FROM HighAvg
                    UNION ALL
                    SELECT * FROM LowAvg;
                   
                    COMMIT;
                    """)


"""Task 7.3: For the weekday selected in (b), please give me a table showing the average demand rate we had at different hours of that weekday throughout 2017. 
Please sort the results in descending order based on the average demand rates."""

conn.executescript("""
                    BEGIN;
                   
                    DROP TABLE IF EXISTS AvgDemandSunday;
                    CREATE TABLE AvgDemandSunday AS
                    SELECT T.weekday, T.hour, AVG(C.demand) AS avg_demand
                    FROM CarSharing AS C
                    INNER JOIN Time AS T ON C.timestamp = T.timestamp
                    WHERE T.weekday = 'Sunday' AND strftime('%Y', C.timestamp) = '2017'
                    GROUP BY T.hour
                    ORDER BY avg_demand DESC;
                   
                    DROP TABLE IF EXISTS AvgDemandMonday;
                    CREATE TABLE AvgDemandMonday AS
                    SELECT T.weekday, T.hour, AVG(C.demand) AS avg_demand
                    FROM CarSharing AS C
                    INNER JOIN Time AS T ON C.timestamp = T.timestamp
                    WHERE T.weekday = 'Monday' AND strftime('%Y', C.timestamp) = '2017'
                    GROUP BY T.hour
                    ORDER BY avg_demand DESC;
                   
                    DROP TABLE IF EXISTS AvgDemandWeekday;
                    CREATE TABLE AvgDemandWeekday AS
                    SELECT * FROM AvgDemandSunday
                    UNION ALL
                    SELECT * FROM AvgDemandMonday;
                   
                    COMMIT;
                    """)


"""Task 7.4: Please tell me what the weather was like in 2017. Was it mostly cold, mild, or hot? Which weather condition (shown in the weather column)
was the most prevalent in 2017? What was the average, highest, and lowest wind speed and humidity for each month in 2017? Please organise this information
in two tables for the wind speed and humidity. Please also give me a table showing the average demand rate for each cold, mild, and hot weather in 2017
sorted in descending order based on their average demand rates."""

conn.executescript("""
                    BEGIN;
                   
                    DROP TABLE IF EXISTS Temp2017;
                    CREATE TABLE Temp2017 AS
                    SELECT T.temp_category, COUNT(T.temp_category) AS count
                    FROM CarSharing AS C
                    INNER JOIN Temperature AS T ON C.id = T.id
                    WHERE strftime('%Y', C.timestamp) = '2017' AND T.temp_category IS NOT NULL
                    GROUP BY T.temp_category
                    ORDER BY count DESC;
                   
                    DROP TABLE IF EXISTS Weather2017;
                    CREATE TABLE Weather2017 AS
                    SELECT W.weather, COUNT(W.weather) AS count
                    FROM CarSharing AS C
                    INNER JOIN Weather AS W ON C.weather_code = W.weather_code
                    WHERE strftime('%Y', C.timestamp) = '2017'
                    GROUP BY W.weather
                    ORDER BY count DESC;
                   
                    DROP TABLE IF EXISTS WindSpeed2017;
                    CREATE TABLE WindSpeed2017 AS
                    SELECT T.month, AVG(C.windspeed) AS avg_wind_speed, MAX(C.windspeed) AS max_wind_speed, MIN(C.windspeed) AS min_wind_speed
                    FROM CarSharing AS C
                    INNER JOIN Time AS T ON C.timestamp = T.timestamp
                    WHERE strftime('%Y', C.timestamp) = '2017'
                    GROUP BY T.month;
                   
                    DROP TABLE IF EXISTS Humidity2017;
                    CREATE TABLE Humidity2017 AS
                    SELECT T.month, AVG(C.humidity) AS avg_humidity, MAX(C.humidity) AS max_humidity, MIN(C.humidity) AS min_humidity
                    FROM CarSharing AS C
                    INNER JOIN Time AS T ON C.timestamp = T.timestamp
                    WHERE strftime('%Y', C.timestamp) = '2017'
                    GROUP BY T.month;
                   
                    DROP TABLE IF EXISTS AvgDemandTemp2017;
                    CREATE TABLE AvgDemandTemp2017 AS
                    SELECT T.temp_category, AVG(C.demand) AS avg_demand
                    FROM CarSharing AS C
                    INNER JOIN Temperature AS T ON C.id = T.id
                    WHERE strftime('%Y', C.timestamp) = '2017' AND T.temp_category IS NOT NULL
                    GROUP BY T.temp_category
                    ORDER BY avg_demand DESC;
                   
                    COMMIT;
                    """)


"""Task 7.5: Give me another table showing the information requested in (d) for the month we had the highest average demand rate
in 2017 so that I can compare it with other months."""

conn.executescript("""
                    BEGIN;
                   
                    DROP TABLE IF EXISTS HighestAvgDemandMonth;
                    CREATE TABLE HighestAvgDemandMonth AS
                    SELECT T.month, AVG(C.demand) AS avg_demand, AVG(C.windspeed) AS avg_wind_speed, MAX(C.windspeed) AS max_wind_speed, 
                    MIN(C.windspeed) AS min_wind_speed, AVG(C.humidity) AS avg_humidity, MAX(C.humidity) AS max_humidity, MIN(C.humidity) AS min_humidity
                    FROM CarSharing AS C
                    INNER JOIN Time AS T ON C.timestamp = T.timestamp
                    WHERE strftime('%Y', C.timestamp) = '2017' 
                    AND T.month = (
                                    SELECT T.month
                                    FROM CarSharing AS C
                                    INNER JOIN Time AS T ON C.timestamp = T.timestamp
                                    WHERE strftime('%Y', C.timestamp) = '2017'
                                    GROUP BY T.month
                                    ORDER BY AVG(C.demand) DESC
                                    LIMIT 1                   
                                    );
                   
                    COMMIT;
                    """)


"""Extracting a pandas dataframe from the database to perform Task 2 in a separate script."""

df = pd.read_sql_query("SELECT * FROM CarSharing_backup", conn)

conn.close()