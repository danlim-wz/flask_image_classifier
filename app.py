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
import torch
import torchvision

#CUDA compatibility
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = InteractiveSession(config=config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
label = 'img'

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

    #display image along with prediction
    if label == 'img':
        image = load_img(image, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        prediction = resnet_model.predict(image)
        prediction = decode_predictions(prediction)[0][0][1]
        prediction = prediction.replace('_',' ')
    else:
        image = load_img(image)
        image = img_to_array(image)
        image = image.transpose()
        image = image / 255
        image = np.expand_dims(image, axis=0)
        with torch.no_grad():
            prediction = model(torch.from_numpy(image).float().to(device))
        image = image * 255
        image = image[0].transpose((2,1,0))
        if label == 'box':
            box = prediction[0]['boxes'][0]
            cv2.rectangle(image, (int(box[3]), int(box[2])), (int(box[1]), int(box[0])), (255,0,0), 2)
            image_pil = Image.fromarray(np.uint8(image))
            image_pil.thumbnail((600,300), Image.ANTIALIAS)
            image_pil.save(file_path)
        elif label == 'mask':
            mask = prediction[0]['masks'][0][0].cpu().numpy().transpose()
            mask[mask > 0.5] = 255
            mask[mask <= 0.5] = 0
            mask = np.array([mask for i in range(3)]).transpose((1,2,0))
            image = cv2.addWeighted(image, 0.5, mask, 0.5, 0)
            image_pil = Image.fromarray(np.uint8(image))
            image_pil.thumbnail((600,300), Image.ANTIALIAS)
            image_pil.save(file_path)
        prediction = COCO_INSTANCE_CATEGORY_NAMES[prediction[0]['labels'][0]]

    return render_template("upload.html", image_path = filename, prediction = 'Prediction: '+prediction)

@app.route("/landing_page",methods=["GET","POST"])
def landing_page():
    return render_template("upload.html", image_path = 'landing_page_pic.jpg')

@app.route("/upload_zip",methods=["GET","POST"])
def upload_zip():
    return render_template("zip.html")

@app.route("/get_img",methods=["GET","POST"])
def get_img():
    global label
    label = 'img'
    return redirect(url_for('landing_page'))

@app.route("/get_box",methods=["GET","POST"])
def get_box():
    global label
    label = 'box'
    return redirect(url_for('landing_page'))

@app.route("/get_mask",methods=["GET","POST"])
def get_mask():
    global label
    label = 'mask'
    return redirect(url_for('landing_page'))

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
    
    COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    """initialize models(replace models to use your own)"""
    #box & mask
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval().cuda()
    with torch.no_grad():
        model(torch.from_numpy(np.random.rand(1,3,224,224)).float().to(device))
    
    #classification
    resnet_model = ResNet50(weights='imagenet')
    resnet_model.predict(np.random.rand(1,224,224,3))

    #start backend
    app.run(host='0.0.0.0', debug=False, threaded=False, port=8000)
