from flask import Flask, render_template, request, redirect
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np
import os
import uuid
import flask
import urllib
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
cors = CORS(app)

# Load crop recommendation model
with open("Crop_recommendation.pickle", 'rb') as f:
    model_recomm = pickle.load(f)

# Load yield prediction model
with open("Yield_prediction.pickle", 'rb') as f:
    model_yield = pickle.load(f)

# Load label encoder
with open('labelencoder.pickle', 'rb') as f:
    label = pickle.load(f)

# Load datasets
csv_yeild = pd.read_csv("Final_csv.csv")
csv_recom = pd.read_csv("labelencoded.csv")
csv_state_names = pd.read_csv('Crop_yield.csv')

len_state = len(sorted(csv_state_names['State_Name'].unique()))


ALLOWED_EXT = set(['jpg', 'jpeg', 'png', 'jfif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXT


classes = ['jute', 'maize', 'rice', 'sugarcane', 'wheat']


def predict(filename, model=None):
    if model is None:
        model = load_model(os.path.join(os.getcwd(), 'CNN_h5_model.h5'))

    img = load_img(filename, target_size=(224, 224), color_mode='grayscale')
    img = img_to_array(img)
    img = img.reshape((1, 224, 224, 1))
    img = np.array(img)

    img = img.astype('float32')
    img = img / 255.0
    result = model.predict(img)

    dict_result = {}
    for i in range(5):
        dict_result[result[0][i]] = classes[i]

    res = result[0]
    res.sort()
    res = res[::-1]
    prob = res[:3]

    prob_result = []
    class_result = []
    for i in range(3):
        prob_result.append((prob[i] * 100).round(2))
        class_result.append(dict_result[prob[i]])

    return class_result, prob_result


@app.route('/', methods=['GET'])
def index():
    return render_template('index1.html')


@app.route('/reccom', methods=['GET'])
def reccom():
    return render_template('reccom.html')


@app.route('/yield', methods=['GET'])
def yield1():
    states = sorted(csv_state_names['State_Name'].unique())
    print("hello")
    print(states)
    print(len(states))
    return render_template('yield.html', states=states)


@app.route('/reccom', methods=['POST'])
@cross_origin()
def reccomend():
    N = request.form.get('n')
    P = request.form.get('p')
    K = request.form.get('k')
    temperature = request.form.get('temp')
    humidity = request.form.get('humidity')
    ph = request.form.get('ph')
    rainfall = request.form.get('rainfall')

    value2 = model_recomm.predict([[N, P, K, temperature, humidity, ph, rainfall]])

    suggest = label.inverse_transform(value2)
    print(suggest)
    return str(suggest[0])


@app.route('/yield', methods=['POST'])
@cross_origin()
def yield_pred():
    area = request.form.get('area')
    year = request.form.get('year')
    rainfall = request.form.get('rainfall')
    select_state = request.form.get('select_state')

    last_of_list = ((sorted(csv_state_names['State_Name'].unique()))[-1])

    if (select_state == (last_of_list)):
        x = np.zeros(len_state + 2)

        x[0] = year
        x[1] = area
        x[2] = rainfall
    else:
        loc_index = np.where(csv_yeild.columns == select_state)[0][0]

        x = np.zeros(len_state + 2)

        loc_index = loc_index - 1
        x[0] = year
        x[1] = area
        x[2] = rainfall
        if loc_index >= 0:
            x[loc_index] = 1

    pred = model_yield.predict([x])[0]
    return str(round(float(pred) / float(area), 2))


@app.route('/home', methods=['GET'])
def home():
    return render_template("home.html")


@app.route('/success', methods=['GET', 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd(), 'static/images')
    if request.method == 'POST':
        if request.form:
            link = request.form.get('link')
            try:
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename + ".jpg"
                img_path = os.path.join(target_img, filename)
                output = open(img_path, "wb")
                output.write(resource.read())
                output.close()
                img = filename

                class_result, prob_result = predict(img_path)

                predictions = {
                    "class1": class_result[0],
                    "class2": class_result[1],
                    "class3": class_result[2],
                    "prob1": prob_result[0],
                    "prob2": prob_result[1],
                    "prob3": prob_result[2],
                }

            except Exception as e:
                print(str(e))
                error = 'This image from this site is not accessible or inappropriate input'

            if len(error) == 0:
                return render_template('success.html', img=img, predictions=predictions)
            else:
                return render_template('home.html', error=error)

        elif request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img, file.filename))
                img_path = os.path.join(target_img, file.filename)
                img = file.filename

                class_result, prob_result = predict(img_path)

                predictions = {
                    "class1": class_result[0],
                    "class2": class_result[1],
                    "class3": class_result[2],
                    "prob1": prob_result[0],
                    "prob2": prob_result[1],
                    "prob3": prob_result[2],
                }

            else:
                error = "Please upload images of jpg, jpeg, and png extension only"

            if len(error) == 0:
                return render_template('success.html', img=img, predictions=predictions)
            else:
                return render_template('home.html', error=error)

    else:
        return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)
