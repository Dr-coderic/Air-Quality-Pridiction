# https://www.dremio.com/resources/tutorials/forecasting-air-quality-with-dremio-python-and-kafka/#toc_item_Intro
# https://openweathermap.org/api/air-pollution

from flask import Flask
from flask import render_template
from flask import request

import tensorflow as tf
import numpy as np
import requests

loaded_model = tf.keras.models.load_model('model')

app = Flask(__name__)

def get_pm2_5(data):
    opn_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={data[0]}&lon={data[1]}&appid=c3dfc12b8f0eac9cface6647f767b0af"
    api_res = requests.get(url=opn_url)
    model_res = None
    pm2_5 = api_res.json()['list'][0]['components']['pm2_5']
    return pm2_5

@app.route("/",  methods=['GET', 'POST'])
def hello_world():
    res = {}
    if request.method == 'POST':
        coordinates = request.form['coordinates'].split(",")
        coordinates = [ float(d) for d in coordinates ]
        model_res = loaded_model.predict(np.array([[get_pm2_5(coordinates)]]))
        res['model_res'] = model_res
        print(model_res)
    return render_template('home.html', data=res)