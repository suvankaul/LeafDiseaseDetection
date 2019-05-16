from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2
from PIL import ImageTk, Image

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
#MODEL_PATH = 'models/model1.h5'
sys.path.append(os.path.abspath("./models"))

# Load your trained model
#model = load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary
#print('Model loaded. Start serving...')

def init():
    json_file = open('./models/model1.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model._make_predict_function() 
    # load weights into new model
    loaded_model.load_weights("./models/model1.h5")
    print("Loaded model from disk")
    
    return loaded_model

global model, graph
model = init()
# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(128, 128))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    label=["Plant: Apple and Disease: Apple scab","Plant: Apple and Disease: Black rot","Plant: Apple and Disease: Cedar_apple_rust","Plant: Apple and Disease: Healthy",
               "Plant: Corn(maize) and Disease: Cercospora leaf spot Gray leaf spot","Plant: Corn_(maize) and Disease: Common rust ",
               "Plant: Corn(maize) and Disease: Healthy","Plant: Corn(maize) and Disease: Northern Leaf Blight","Plant: Grape and Disease: Black rot",
               "Plant: Grape and Disease: Esca(Black Measles)","Plant: Grape and Disease: Healthy","Plant: Grape and Disease: Leaf blight (Isariopsis Leaf Spot)",
               "Plant: Potato and Disease: Early blight","Plant: Potato and Disease: Healthy","Plant: Potato and Disease: Late blight","Plant: Tomato and Disease: Bacterial spot",
               "Plant: Tomato and Disease: Early blight","Plant: Tomato and Disease: Healthy","Plant: Tomato and Disease: Late blight","Plant: Tomato and Disease: Leaf Mold",
               "Plant: Tomato and Disease: Septoria leaf spot","Plant: Tomato and Disease: Spider mites Two-spotted spider mite","Plant: Tomato and Disease: Target Spot",
               "Plant: Tomato and Disease: Tomato Yellow Leaf Curl Virus","Plant: Tomato and Disease: Tomato mosaic virus"]
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')
    preds = model.predict(x)
    label2=label[preds.argmax()]
    print(label2)
    return label2


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/index.html', methods=['GET'])
def index1():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        ##pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        ##result = str(pred_class[0][0][1])               # Convert to string
        result = str(preds)
        return result
    return None


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

    # Serve the app with gevent
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()
