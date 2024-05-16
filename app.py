import os
import uuid
from flask import Flask, render_template, request, jsonify, url_for
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__, template_folder='templates')


PLOT_FOLDER = os.path.join('static', 'plots')
os.makedirs(PLOT_FOLDER, exist_ok=True)


data = pd.read_csv('dataset/punedataset2.csv')

label_encoder = LabelEncoder()
data['Location_Code'] = label_encoder.fit_transform(data['Location'])


X = data[['Area', 'Bedrooms', 'Year', 'Location_Code']].values
y = data['Price'].values


model = LinearRegression()
model.fit(X, y)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    area = float(request.form['area'])
    bedrooms = float(request.form['bedrooms'])
    year = float(request.form['year'])
    location = request.form['location']

    location_code = label_encoder.transform([location])[0]

    features = np.array([[area, bedrooms, year, location_code]])

    prediction_value = model.predict(features)[0]
    price_per_sqft = prediction_value / area

    plot_filename = f"plot_{uuid.uuid4()}.png"
    plot_path = os.path.join(PLOT_FOLDER, plot_filename)
    plot_path = plot_path.replace('\\', '/')
    plt.figure()
    plt.title("Prediction Plot")
    plt.xlabel("Area")
    plt.ylabel("Price")


    plt.scatter(data['Area'], data['Price'], color='blue', label='Actual Prices')


    plt.scatter(area, prediction_value, color='red', label='Predicted Price')
    plt.legend()
    plt.savefig(plot_path)
    plt.close()

    plot_url = url_for('static', filename=plot_path.split('static/')[-1])

    return jsonify({
        'location':location,
        'prediction': float(prediction_value),
        'price_per_sqft': float(price_per_sqft),
        'plot': plot_url
    })


if __name__ == '__main__':
    app.run(debug=True)
