from flask import Flask, request, make_response, render_template, redirect, url_for, send_file
import keras
import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array
import os
from PIL import Image
from io import BytesIO, StringIO
from zipfile import ZipFile
import cv2
import numpy as np

#CUDA compatibility
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

app = Flask(__name__)

#set paths to upload folder
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['IMAGE_UPLOADS'] = os.path.join(APP_ROOT, 'static')


@app.route("/display",methods=["GET","POST"])
def display():
    #read and upload resized files to folder
    image = request.files['input_file']
    filename = image.filename
    file_path = os.path.join(app.config["IMAGE_UPLOADS"], filename)
    image_pil = Image.open(image)
    image_pil.thumbnail((600,300), Image.ANTIALIAS)
    image_pil.save(file_path)

    #classify image
    image = load_img(image, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    prediction = resnet_model.predict(image)
    prediction = decode_predictions(prediction)[0][0][1]
    prediction = prediction.replace('_',' ')
    return render_template("upload.html", image_path = filename, prediction = 'Prediction: '+prediction)

@app.route("/landing_page",methods=["GET","POST"])
def landing_page():
    return render_template("upload.html", image_path = 'landing_page_pic.jpg')

@app.route("/upload_zip",methods=["GET","POST"])
def upload_zip():
    return render_template("zip.html")

@app.route("/label_images",methods=["GET","POST"])
def label_images():
    images = request.files['input_file']
    data = BytesIO()
    with ZipFile(images, mode='r') as input_zip:
        with ZipFile(data, mode='a') as image_data:
            for name in input_zip.namelist():
                base_name = os.path.basename(name)
                try:
                    if name.endswith(('.png', '.jpg', '.jpeg')):
                        new_name,ext = os.path.splitext(base_name)
                        img = cv2.imdecode(np.frombuffer(input_zip.read(name), dtype=np.uint8), flags=1)
                        #img = img[:,:,:3]
                        image = cv2.resize(img, (224, 224))
                        image = img_to_array(image)
                        image = np.expand_dims(image, axis=0)
                        image = preprocess_input(image)
                        prediction = resnet_model.predict(image)
                        prediction = decode_predictions(prediction)[0][0][1]
                        prediction = prediction.replace('_',' ')
                        _, encoded_img = cv2.imencode(ext, img)
                        image_data.writestr(prediction+'/'+base_name, encoded_img)
                except:
                    pass
    data.seek(0)
    return send_file(data, attachment_filename='output.zip', as_attachment=True)

if __name__ == '__main__':
    resnet_model = ResNet50(weights='imagenet')
    resnet_model.predict(np.random.rand(1,224,224,3))
    app.run(host='0.0.0.0', debug=False, threaded=False, port=8000)
