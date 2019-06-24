from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import torch
from torch import nn
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/classifier.h5'
classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

decoder = {}
for i in range(len(classes)):
    decoder[classes[i]] = i
encoder = {}
for i in range(len(classes)):
    encoder[i] = classes[i]

    
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.resnet =  EfficientNet.from_name('efficientnet-b0')
        self.l1 = nn.Linear(1000 , 256)
        self.dropout = nn.Dropout(0.75)
        self.l2 = nn.Linear(256,6)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.resnet(input)
        x = x.view(x.size(0),-1)
        x = self.dropout(self.relu(self.l1(x)))
        x = self.l2(x)
        return x
classifier = Classifier()
classifier.load_state_dict(torch.load(MODEL_PATH,map_location=lambda storage, loc: storage))

classifier.eval()

transform = transforms.Compose([        transforms.ToPILImage(),
                                        transforms.Resize((150,150)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.43018365, 0.45747966, 0.45386454],[0.23611858, 0.23468056, 0.2432306 ])])

def im_convert(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image,(150,150))
    image = transform(image)
    image = image.view((1,3,150,150))
    return image


def model_predict(image_path):
    image = im_convert(image_path)
    image = image.type(torch.FloatTensor)
    pred = model(image)
    pred = np.argmax(pred.detach().numpy())
    return encoder[int(pred)]


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.getcwd()
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result = model_predict(file_path)

        return str(result)
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
