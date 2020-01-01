# Web app for image labelling

The following repository contains a web app for users to label their raw image data, expediting the process of data cleaning and labelling.

Users may upload a zip file where the app will then sort and return the images in folders titled with the predicted image class based on the model utilized.

Using your own pretrained model is as simple as importing it and changing a single line of code in the *app.py* file.

![web_app](https://user-images.githubusercontent.com/52344837/70791925-b544b680-1dca-11ea-8386-d1cc04939981.gif)

# Usage
1) Clone/download this repository
2) Install dependencies
``` 
pip install -r requirements.txt
```
3) Launch the web app
```
python app.py
```
4) Open your browser and head over to **localhost:8000/landing_page** and begin using the app

Alternatively, try it out with Docker: 

**docker run -d -p 8000:8000 danlimwz/flask_image_classifier**

# Supported classes

The model uses a ResNet50 architecture which was pretrained on the ImageNet dataset. Visit this link for supported image classes: http://image-net.org/explore

# Implementation

Check out the article I wrote on medium for a tutorial on how to create the app: https://medium.com/@limwz.daniel/deploying-your-deep-learning-model-using-flask-and-docker-c05a6d1d96a5 

# Future extensions
1) Image class selection
2) Confidence threshold selection
3) Facial recognition
