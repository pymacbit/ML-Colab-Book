#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 14:51:03 2019

@author: pymacbit
"""

# Importing libraries
from __future__ import print_function
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential


# Read the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sub = pd.read_csv('gender_submission.csv')


#Data Preprocessing/Cleaning
#Convert Sex of Male/Female to 1/0
train['Sex'].replace(['male', 'female'], [1, 0], inplace=True)
test['Sex'].replace(['male','female'],[1,0],inplace=True)

#Convert Embarked place C/Q/S to 0/1/2
train['Embarked'].replace(['C','Q','S'],[0,1,2], inplace=True)
test['Embarked'].replace(['C','Q','S'],[0,1,2], inplace = True)

train['Cabin'] = train['Cabin'].str.extract('(\d+)', expand=True)
test['Cabin'] = test['Cabin'].str.extract('(\d+)', expand=True)

train = train.fillna(0)
test = test.fillna(0)

y_train = train['Survived']
testName = test['PassengerId']
train.drop(['PassengerId','Survived','Name','Ticket'],inplace=True,axis=1)
test.drop(['PassengerId','Name','Ticket'],inplace=True,axis=1)


#ANN model
model = Sequential()
model.add(Dense(16, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(14, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(6, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile and Fit the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train, y_train, epochs=200, batch_size=10,  verbose=2)

# calculate predictions
predictions = model.predict(test)

# round predictions
rounded = [int(round(x[0])) for x in predictions]

#print(testName.to_frame())
output = pd.concat([testName.to_frame(),pd.DataFrame(rounded)],axis=1)
output.columns = ['PassengerId','Survived']
# print(output)
output.to_csv('submission.csv',index=False)