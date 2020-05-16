import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

baseDirectory = os.getcwd()


def clean_raw_data():
    # does not run, but here to show how data was organized
    # gets the mean temperature and snow (cm) per day
    rawDataDailyLocation = baseDirectory + "\\rawDaily.csv"
    rawDataDaily = pd.read_csv(rawDataDailyLocation, encoding='utf-8-sig', low_memory=False)
    uselessCol = ['Longitude', 'Latitude', 'Station Name', 'Climate ID',
                  'Date/Time', 'Year', 'Month', 'Day', 'Data Quality', 'Max Temp',
                  'Max Temp Flag', 'Min Temp', 'Min Temp Flag',
                  'Mean Temp Flag', 'Heat Deg Days', 'Heat Deg Days Flag',
                  'Cool Deg Days', 'Cool Deg Days Flag', 'Total Rain',
                  'Total Rain Flag', 'Total Snow Flag', 'Total Precip',
                  'Total Precip Flag', 'Snow on Grnd', 'Snow on Grnd Flag',
                  'Dir of Max Gust', 'Dir of Max Gust Flag', 'Spd of Max Gust',
                  'Spd of Max Gust Flag'
                  ]
    rawDataDaily.drop(uselessCol, axis=1, inplace=True)

    # calculates the mean humidity and wind speed
    rawDataHourlyLocation = baseDirectory + "\\rawHourly.csv"
    rawDataHourly = pd.read_csv(rawDataHourlyLocation, encoding='utf-8-sig', low_memory=False, usecols=[14, 18])

    hourlyHumidity, hourlyWindSpd = 0, 0
    relHum, windSpd = [], []
    rawDataHourly['Rel Hum'].fillna(rawDataHourly['Rel Hum'].mean(), inplace=True)
    rawDataHourly['Wind Spd'].fillna(rawDataHourly['Wind Spd'].mean(), inplace=True)

    for i in range(1, 43825):
        hourlyHumidity += float(rawDataHourly.iat[i, 0])
        hourlyWindSpd += float(rawDataHourly.iat[i, 1])
        if i % 24 == 0:
            iToArray = round(i / 24) - 1
            relHum.append(iToArray)
            windSpd.append(iToArray)
            relHum[iToArray] = round(hourlyHumidity / 24)
            windSpd[iToArray] = round(hourlyWindSpd / 24)
            hourlyHumidity, hourlyWindSpd = 0, 0

    # adds the humidity and wind speed
    rawDataDaily.insert(1, "Rel Hum", relHum, True)
    rawDataDaily.insert(2, "Wind Spd", windSpd, True)
    # Used for regression, if it snows, then calculates how much it snowed
    rawDataDaily.to_csv("cleanRegression.csv")
    rawDataDaily.loc[rawDataDaily['Total Snow'] > 0, 'Total Snow'] = 1
    # Used for classification, determines if it snowed or not
    rawDataDaily.to_csv("cleanClassification.csv")


def snow_boolean():
    cleanBooleanLocation = baseDirectory + "\\cleanClassification.csv"
    cleanBoolean = pd.read_csv(cleanBooleanLocation, encoding='utf-8-sig', low_memory=False)
    cleanBoolean['Total Snow'].fillna(cleanBoolean['Total Snow'].mean(), inplace=True)

    variables = cleanBoolean.drop('Total Snow', axis=1)
    value = cleanBoolean['Total Snow']
    X_train, X_test, y_train, y_test = train_test_split(variables, value, test_size=0.4, random_state=42)

    X_train.to_csv("boolTrainFeatures.csv", index=False)
    X_test.to_csv("boolTestFeatures.csv", index=False)
    y_train.to_csv("boolTrainLabels.csv", index=False)
    y_test.to_csv("boolTestLabels.csv", index=False)

    trainVariables = pd.read_csv('boolTrainFeatures.csv')
    trainLabels = pd.read_csv('boolTrainLabels.csv')

    rf = RandomForestClassifier()
    # scores = cross_val_score(rf, trainVariables, trainLabels.values, cv=5)
    # print(scores)

    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)

    print("Enter the temperature, humidity and wind speed to see if it snows")

    var = input("Temperature (°C): ")
    temperature = float(var)
    var = input("Humidity(%): ")
    humidity = float(var)
    var = input("Wind Speed(km/h): ")
    windSpeed = float(var)

    inputValues = [[0, temperature, windSpeed, humidity]]
    boolValue = float(rf.predict(inputValues)[0])

    regressionValue = float(snow_amount(temperature, humidity, windSpeed))

    print(boolValue)
    print(regressionValue)

    if boolValue == 0.0 and regressionValue < 0.5:
        print("Given the following data, it will not snow")
    elif (boolValue == 1.0 and regressionValue < 0.5) or (boolValue == 0.0 and regressionValue > 0.5):
        print("It might snow. If it does, it would be under 1cm")
    else:
        print("It will snow and should be around " + str(regressionValue) + " cm.")


def snow_amount(temperature, humidity, windSpeed):
    cleanRegressionLocation = baseDirectory + "\\cleanRegression.csv"
    cleanRegression = pd.read_csv(cleanRegressionLocation, encoding='utf-8-sig', low_memory=False)

    variables = cleanRegression.drop('Total Snow', axis=1)
    value = cleanRegression['Total Snow']
    X_train, X_test, y_train, y_test = train_test_split(variables, value, test_size=0.4, random_state=42)

    X_train.to_csv("regressionTrainFeatures.csv", index=False)
    X_test.to_csv("regressionTestFeatures.csv", index=False)
    y_train.to_csv("regressionTrainLabels.csv", index=False)
    y_test.to_csv("regressionTestLabels.csv", index=False)

    clf = LinearRegression()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    inputValues = [[temperature, windSpeed, humidity]]
    regressionValue = float(clf.predict(inputValues)[0])

    return regressionValue


if __name__ == "__main__":
    snow_boolean()
