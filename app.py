import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    For rendering results on HTML GUI
    """
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    mapper = {0: 'Employee will stay', 1: 'Employee will leave'}
    prediction = pd.Series(prediction).map(mapper)
    output = prediction[0]
    return render_template('index.html', prediction_text='Model Prediction : {0}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
