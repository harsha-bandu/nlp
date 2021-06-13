# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 23:10:42 2021

@author: harsha
"""

from flask import Flask, request, jsonify, render_template

import numpy as np
#import pandas as pd
import pickle

app = Flask(__name__)
filename = 'spam_model.pkl'
model = pickle.load(open(filename, 'rb'))

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    
    arr = np.array([[data1,data2]])
    pred = model.predict(arr)
    
    return render_template('after.html', data = pred)

if __name__ == "__main__":
    app.run(debug = True)