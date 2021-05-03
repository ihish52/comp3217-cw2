import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

#Reading training data
trainDF = pd.read_csv('TrainingData.txt', header=None)
y = trainDF[24].tolist()
trainDF = trainDF.drop(24, axis=1)
x = trainDF.values.tolist()
#x_classify_2 = x[4500:5501]
#print(x_classify_2)

#Reading testing data to predict
testDF = pd.read_csv('TestingData.txt', header=None)
x_classify = testDF.values.tolist()

#Splitting training data for testing algorithm
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
#x_test_copy = x_test

#Scaling between 0 and 1 - normalising
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_classify = scaler.transform(x_classify)
#x_classify_2 = scaler.transform(x_classify_2)

#Logistic Regression
log = linear_model.LogisticRegression(random_state = 0,solver = 'liblinear',multi_class = 'auto')
log = log.fit(x_train, y_train)
y_pred = log.predict(x_classify)
y_pred = [int(x) for x in y_pred]

#Checking predictions on 25% 'testing' dataset
#y_test_pred = log.predict(x_classify_2)
#for i in y_test_pred:
#    print(int(i), end = ',')
#    
#print(x_classify_2[0:5])

#Testing and training scores
print("Training accuracy:",log.score(x_train, y_train))
print("Testing accuracy for 25% of training data:",log.score(x_test, y_test))

#Printing results to output file
predDF = pd.DataFrame({'Prediction': y_pred})
testDF = testDF.join(predDF)
testDF.to_csv("TestingDataOutput.txt", header=None, index=None)
predDF.to_csv("PredictionsOnly.txt", header=None, index=None)