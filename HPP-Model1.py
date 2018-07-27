#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 13:00:24 2018

@author: guidosidoni
"""

import pandas as pd	
import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers
from keras import metrics

testing = pd.read_csv('/Users/guidosidoni/Documents/NEU/Misselanea/HousePricePrediction/train.csv')

testing.columns
testing.fecha

########## CREATION OF NEW VARIABLES #################
testing['precioM2'] = testing.precio / testing.metrostotales
testing['FechaD'] = pd.to_datetime(testing.fecha, format="%Y-%m-%d")

#add year, month

testing.describe()

##### Looking for NA #######
testing.isnull().values.any()
testing.isnull().any()

testing.titulo
testing.descripcion
testing.tipodepropiedad
testing['tipodepropiedad'] = testing['tipodepropiedad'].fillna(' ')

testing.direccion
testing.ciudad
crosstabCiudad = pd.crosstab(testing.ciudad,1)
crosstabCiudad = crosstabCiudad.rename(columns={1:'count'})
crosstabCiudad.sort_values('count',ascending=False)
testing['ciudad'] = testing['ciudad'].fillna(' ')

testing.provincia
pd.crosstab(testing.provincia, 1).rename(columns={1:'count'}).sort_values('count', ascending=False)
testing['provincia'] = testing['provincia'].fillna(' ')

testing.antiguedad
pd.crosstab(testing.habitaciones, 1)
testing['habitaciones'] = testing['habitaciones'].fillna(0)

testing['gimnasio'] = testing['gimnasio'].fillna(0)
testing['usosmultiples'] = testing['usosmultiples'].fillna(0)
testing['garages'] = testing['garages'].fillna(0)
testing['banos'] = testing['banos'].fillna(0)
testing['metroscubiertos'] = testing['metroscubiertos'].fillna(0)
testing['metrostotales'] = testing['metrostotales'].fillna(0)


##ploting price by year
year = testing.groupby(testing.FechaD.dt.year )
priceYear = (year.agg({'precio':'mean', 'precioM2':'mean'}) 
    .rename(columns={'precio':'mean_precio', 'precioM2':'mean_precioM2'}))

plt.plot(priceYear.index, priceYear.mean_precio, '-o')
plt.plot(priceYear.index, priceYear.mean_precioM2, '-o')


#ploting price by month and year
monthyear = testing.set_index('FechaD')
monthyear = monthyear.resample('M').mean()
plt.plot(monthyear.index, monthyear.precio, '-o')
plt.plot(monthyear.index, monthyear.precioM2, '-o')

############################################

####One hot encoding
testingShort = testing[['tipodepropiedad','provincia','precio','habitaciones','garages','banos',
                       'metroscubiertos','metrostotales', 'gimnasio','usosmultiples','piscina',
                       'escuelascercanas','centroscomercialescercanos']]


testingShort2 = pd.concat([testingShort.iloc[:,2:], 
                         pd.get_dummies(testingShort.provincia), 
                         pd.get_dummies(testing.tipodepropiedad)
                         ], axis=1)

testingShort2.columns

##### Shuffle data
testingShort2 = np.array(testingShort2 )
np.random.shuffle(testingShort2)
testingShort2
testingShort2.shape
##### Standarized Values

train_data = testingShort2[:,1:]
train_target = testingShort2[:,0]

mean = train_data.mean(axis=0)
mean.shape
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std


######### MODEL SETING ##########

def build_model(train_data):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=[metrics.mean_squared_logarithmic_error])
    return model


train_data, val_data, train_targets, val_targets = train_test_split(train_data, train_target, random_state = 222, test_size=0.2)

model = build_model(train_data)

model.summary()

#we save the history of the fiting.
history = model.fit(train_data, 
                    train_targets, 
                    validation_data=(val_data, val_targets),
                    epochs=100, batch_size=1, 
                    verbose=1)

#SAVE THE MODEL
model.save('HPP_model1.h5')

#PLOTING THE OUTPUT

def plotResults():
    acc = history.history['mean_squared_logarithmic_error']
    val_acc = history.history['val_mean_squared_logarithmic_error']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc)+1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Valiudation acc')     
    plt.title('Training and validation accuracy')   
    plt.legend() 

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

plotResults()





