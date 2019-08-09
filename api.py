#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 12:02:33 2019

@author: anu
"""
from flask import Flask,jsonify,request
from flask_cors import CORS
import cv2
import numpy as np
from skimage.transform import rescale
from keras.models import load_model
from keras.preprocessing import image



app = Flask(__name__)
cors = CORS(app)
@app.route('/',methods=['GET'])
def get():
    return "hello"    

@app.route('/predict',methods=['POST'])
def upload_image():    
    filestr = request.files['file'].read()
    npimg = np.fromstring(filestr, np.uint8)
    img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    print('imdecode',img)
    img = img[...,::-1]
    print('test_image',img)
   
    img = rescale(img, 1./255, anti_aliasing=False)
   
    img = cv2.resize(img, (700,460))
    
    img = np.expand_dims(img, axis = 0)
   
    
    model=load_model('my_model.h5')
    result = model.predict(img)
    print(result)
    result_round = np.round(result,3)
    print(result_round[0][0])
    return jsonify({"result": result_round[0][0]*100})
    
   
    #return jsonify({})
    

app.run(port=8098)

'''model = load_model('my_model.h5')

# Get test image ready
test_image = image.load_img('../data_only/fold1/testing/0/SOB_B_TA-14-3411F-100-014_class0.png', target_size=(460,700,3))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

#test_image = test_image.reshape(460,700,3)    # Ambiguity!
# Should this instead be: test_image.reshape(img_width, img_height, 3) ??

result = model.predict(test_image, batch_size=1)
print (result)'''