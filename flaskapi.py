#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 13:25:46 2020

@author: Sacha Hamiche
"""

from flask import Flask, render_template, jsonify
app = Flask(__name__)
import pickle
import pandas as pd
import numpy as np
import matplotlib, seaborn, bokeh
from os import listdir
from os.path import isfile, join
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
datafiles = [f for f in listdir('data') if isfile(join('data', f))]
def splitdata(data,value):
        splited = np.split(data, value)
        temp =  []
        for v in splited:
            temp.append(np.mean(v))
        return temp
dffinal = pd.DataFrame()
for k in datafiles:
    print(k +" loading...")
    with open('data/'+k, 'rb') as f:
        data = pickle.load(f, encoding="latin1")
    #On supprime les données jugé non pertinente
    del data['signal']['chest']['EDA']
    del data['signal']['chest']['Temp']
    del data['signal']['chest']['EMG']
    #On stock le questionnaire dans une variable et on le supprime du dict
    quest = data['questionnaire']
    del data['questionnaire']
    #Initialise dataframe with acivity
    df = pd.DataFrame(data["activity"], columns=['activity'])
    for j,v in quest.items():
        df[j] = v
    df["Subject"] = k.split(".")[0]
    rpeak = [0] * len(data['signal']['chest']['ACC'])
    #Trop lent
    #for v in range(len(rpeak)):
    #    if v+1 in data["rpeaks"]:
    #        v=1
    #Plus rapide
    for v in data["rpeaks"]:
        rpeak[v-1] = 1
    newlabel = np.repeat(data["label"], 8)

    diff = len(data["activity"])-len(newlabel)
    for i in range(diff):
        newlabel = np.append(newlabel,newlabel[-1])

    #4HZ mesure
    df['signal.wrist.EDA'] = data['signal']['wrist']['EDA']
    df['signal.wrist.TEMP'] = data['signal']['wrist']['TEMP']
    #other mesure
    df['label'] = splitdata(newlabel,len(data['activity']))
    df['rpeak'] = splitdata(np.asarray(rpeak),len(data['activity']))
    df['signal.wrist.BVP'] = splitdata(data['signal']['wrist']['BVP'],len(data['activity']))
    df['signal.wrist.ACC'] = splitdata(data['signal']['wrist']['ACC'],len(data['activity']))
    df['signal.chest.ACC'] = splitdata(data['signal']['chest']['ACC'],len(data['activity']))
    df['signal.chest.ECG'] = splitdata(data['signal']['chest']['ECG'],len(data['activity']))
    df['signal.chest.Resp'] = splitdata(data['signal']['chest']['Resp'],len(data['activity']))
    dffinal = dffinal.append(df, ignore_index=True)
    print(str(k) +" loaded. Shape of final DataFrame: "+ str(dffinal.shape))
labels = dffinal['Gender'].unique().tolist()
mapping = dict( zip(labels,range(len(labels))) )
dffinal.replace({'Gender': mapping},inplace=True)

labels = dffinal['Subject'].unique().tolist()
mapping = dict( zip(labels,range(len(labels))) )
dffinal.replace({'Subject': mapping},inplace=True)
print("Data loading ended.")

y = dffinal['activity']
x = dffinal.drop(columns=['activity'])
feature = x.columns
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 25)
print("Data preprocessing end")
rf = RandomForestClassifier()
rf.fit(xTrain, yTrain)
print("Predicting...")
rf_predictions = rf.predict(xTest)
print("Prediction done.")
@app.route('/randomForest')
def index():
    return jsonify(yactuel=yTest.tolist(), ypred=rf_predictions.tolist())
print("API is ready")
if __name__ == '__main__':
    app.run(debug=True)
