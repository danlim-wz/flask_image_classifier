# Flask Image Classifier

The following repo is a simple web app for users to label their raw image data, expediting the process of data cleaning and labelling.

Users may upload a zip file where the app will then sort and return the images in folders titled with the prediction of the image class.

The model uses a ResNet50 architecture which was pretrained on the ImageNet dataset. Visit this link for supported image classes: http://image-net.org/explore

Check out the article I wrote on medium: https://medium.com/@limwz.daniel/deploying-your-deep-learning-model-using-flask-and-docker-c05a6d1d96a5 

Try it out with Docker: docker run -d -p 8000:8000 danlimwz/flask_image_classifier

![flask_image_classifier](https://user-images.githubusercontent.com/52344837/61465768-4abe5480-a9ab-11e9-88fb-46fa543fd8e7.gif)
![web_app](https://user-images.githubusercontent.com/52344837/70791925-b544b680-1dca-11ea-8386-d1cc04939981.gif)
