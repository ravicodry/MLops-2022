#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 22:21:54 2022

@author: ravi
"""

from flask import Flask
from flask import request
from joblib import load

app = Flask(__name__)
model_path = "svm_gamma=0.001_C=0.5.joblib"
model = load(model_path)
@app.route("/")
def hello_world():
    return "<!-- hello --> <b> Hello, World!</b>"


# get x and y somehow    
#     - query parameter
#     - get call / methods
#     - post call / methods ** 

# @app.route("/sum", methods=['POST'])
# def sum():
#     x = request.json['x']
#     y = request.json['y']
#     z = x + y 
#     return {'sum':z}



@app.route("/predict", methods=['POST'])
def predict_digit():
    image1 = request.json['image1']
    image2 = request.json['image2']
    predicted1 = model.predict([image1])
    predicted2 = model.predict([image2])
    is_same = False
    if predicted1[0]==predicted2[0]:
        is_same=True
    return (str(is_same))

if __name__=="__main__":
    app.run(host='0.0.0.0',port=5000)