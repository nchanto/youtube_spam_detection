from flask import Flask, url_for, render_template, request
from logging import debug
import pickle
import joblib
import pandas as pd
import numpy as np

app=Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    nb = open('data/naive_bayes.pkl', 'rb')
    naive_bayes = joblib.load(nb)
    count_vect = open('data/vectorizer.pkl', 'rb')
    cv = joblib.load(count_vect)

    if request.method =='POST':
        comment = request.form['comment']
        com =[comment]
        vect = cv.transform(com).toarray()
        pred = naive_bayes.predict(vect)
        
    if pred == 1:
        return render_template("spam.html")
    else:
        return render_template("ham.html")

if __name__ == '__main__':
    app.run(port=8000, debug=True)