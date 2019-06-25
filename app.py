from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import torch
from torch import nn
import cv2
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from torchvision import transforms
import matplotlib.pyplot as plt
from efficientnet_pytorch import  EfficientNet
from torchvision import models

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/classifier1.h5'
classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

decoder = {}
for i in range(len(classes)):
    decoder[classes[i]] = i
encoder = {}
for i in range(len(classes)):
    encoder[i] = classes[i]
sm = nn.Softmax()

def prediction_bar(output,encoder):
    output = output.cpu().detach().numpy()
    a = output.argsort()
    a = a[0]

    size = len(a)
    if(size>5):
        a = np.flip(a[-5:])
    else:
        a = np.flip(a[-1*size:])
    prediction = list()
    clas = list()
    for i in a:
      prediction.append(float(output[:,i]*100))
      clas.append(str(i))
    cl = list()
    for i in a:
        cl.append(encoder[int(i)])
    plt.bar(cl,prediction)
    plt.title("Confidence score bar graph")
    plt.xlabel("Confidence score")
    plt.ylabel("Class number")
    plt.savefig('templates/pred_bar.jpg')
    
class classifie(nn.Module):
    def __init__(self):
        super(classifie, self).__init__()
        model = models.densenet121()
        model = model.features
        #model = EfficientNet.from_pretrained('efficientnet-b3')
        #model =  nn.Sequential(*list(model.children())[:-3])
        self.model = model
        self.linear = nn.Linear(2048, 512)
        self.bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.2)
        self.elu = nn.ELU()
        self.out = nn.Linear(512, 6)
        self.bn1 = nn.BatchNorm1d(2048)
        self.dropout2 = nn.Dropout(0.2)
    def forward(self, x):
        out = self.model(x)
        avg_pool = nn.functional.adaptive_avg_pool2d(out, output_size = 1)
        max_pool = nn.functional.adaptive_max_pool2d(out, output_size = 1)
        out = torch.cat((avg_pool,max_pool),1)
        batch = out.shape[0]
        out = out.view(batch, -1)
        conc = self.linear(self.dropout2(self.bn1(out)))
        conc = self.elu(conc)
        conc = self.bn(conc)
        conc = self.dropout(conc)
        res = self.out(conc)
        return res    

classifier = classifie()
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
    pred = classifier(image)
    return sm(pred)



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
        pred = model_predict(file_path)
        result = np.argmax(pred.detach().numpy())
        result = encoder[int(result)]
        #prediction_bar(pred,encoder)
        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
