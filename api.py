import flask
from flask import Flask, request, render_template
from sklearn.externals import joblib
import numpy as np
from scipy import misc
import torch
import os
import cv2
import wget
wget.download('http://bit.ly/malariaweight')
app = Flask(__name__)

model = torch.load('malaria.h5' , map_location='cpu')
@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method=='POST':
        file = request.files['image']
        if not file: return render_template('index.html', label="No file")
        img = cv2.imread(file)
        img = cv2.resize(img,(150,150))/255
        img = np.reshape(img,(1,3,150,150))
        print(image.shape)
        # make prediction on new image
        img = torch.from_numpy(img)
        img = img.type(torch.FloatTensor)
        prediction = model(img)
        label = str(np.squeeze(prediction))

        # switch for case where label=10 and number=0
        return render_template('index.html', label=label)


if __name__ == '__main__':
	# load ml model
	model = joblib.load('model.pkl')
	# start api
	app.run(host='0.0.0.0', port=8000, debug=True)
