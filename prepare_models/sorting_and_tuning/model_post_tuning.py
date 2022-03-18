# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
import fileinput
import sys
import os

# Select column: Low, High, Open or Close
column='Close'
# These are only testing dates.
startDate='2020-06-27'
endDate='2020-11-25'
instrument = 'GBPUSD=X'
days_into_accountValues = [30]
valueRequiredForSaving = 0.0057
pathToOutput = 'models_sort_output/'

scaler = MinMaxScaler(feature_range=(0,1))
x_test = []
y_test = []

def prepareData(days_into_account):

    global column
    global startDate
    global endDate
    global instrument

    global x_test
    global y_test

    global scaler

    x_test = []
    y_test = []

    # Get the data
    df = web.DataReader(instrument, data_source='yahoo', start=startDate, end=endDate)

    # Create a new dataframe with only our column
    data = df.filter([column])

    # Convert the dataframe to Numpy array
    dataset = data.values

    # Scale the data
    scaled_data = scaler.fit_transform(dataset)

    # Create the testing data set
    test_data = scaled_data

    # Create the data sets x_test and y_test
    y_test = dataset[days_into_account:, :]

    for i in range(days_into_account, len(test_data)):
      x_test.append(test_data[i-days_into_account:i, 0])

    # Convert the data into Numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# ----------------------------------
# Main function

# 1. Load models
models = list()
pathToModel = 'models_to_be_sorted_and_tuned/'
modelsNames = os.listdir(pathToModel)
modelsNames.sort()

for modelName in modelsNames:
    if os.path.isdir(pathToModel + modelName):
        filename = pathToModel + modelName
        model = keras.models.load_model(filename)
        models.append(model)

# 2. Prepare some 2D array for all results
allResults = []

# 3. Prepare data for every days_into_account value
for days_into_accountValue in days_into_accountValues:
    thisDaysValueResults = []
    prepareData(days_into_accountValue)

    # 3a. Get the predictions
    predictions = [model.predict(x_test) for model in models]
    scaledPredictions = []
    for prediction in predictions:
        scaledPrediction = scaler.inverse_transform(prediction)
        scaledPredictions.append(scaledPrediction)

    for prediction in scaledPredictions:
        average = np.average(np.abs(prediction - y_test))
        thisDaysValueResults.append(average)
        # print("AVERAGE ERROR:")
        # print(average)
        # print()

    allResults.append(thisDaysValueResults)

# Convert the data into Numpy array
allResults = np.array(allResults)
allResults = allResults.transpose()

print("")
print("All results:")
print(allResults)
print("")

# Find the lowest value per each model (category index)
modelsCategoriesIndices = []
for result in allResults:
    lowestValue = np.argmin(result)
    modelsCategoriesIndices.append(lowestValue)

print("")
print("Models Categories Indices:")
print(modelsCategoriesIndices)
print("")

for index, model in enumerate(models):
    correctCategoryIndex = modelsCategoriesIndices[index]
    bestResult = allResults[index][correctCategoryIndex]
    if (bestResult < valueRequiredForSaving):
        path = pathToOutput + instrument + '/' + column + '/' + str(days_into_accountValues[correctCategoryIndex]) + '/'
        model.save(path + str(bestResult))
        print("")
        print("Model saved into:")
        print(path)
        print("Model best result:")
        print(bestResult)
        print("")
    else:
        path = pathToOutput + instrument + '/' + column + '/' + 'trash/' + str(days_into_accountValues[correctCategoryIndex]) + '/'
        model.save(path + str(bestResult))
        print("")
        print("Model saved into trash.")
        print("Model best result:")
        print(bestResult)
        print("")


    # 3. Calculate average
    # meanPrediction = np.mean(scaledPredictions, axis=0)

    # 4. Compare results
    # print("AVERAGE ENSEBLE ERROR:")
    # averageEnsemble = np.average(np.abs(meanPrediction - y_test))
    # print(averageEnsemble)
    # print()

    # np.savetxt("average_error_data.csv", np.abs(meanPrediction - y_test), delimiter=",")


# prepareData()
