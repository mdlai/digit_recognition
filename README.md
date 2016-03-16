### Prerequisites and Instructions
The notebooks use the MNIST data and classify it with KNN and CNN.

To launch Tensorboard navigate to the tensor_flow_logs folder and use:
```
python ~/path/to/python/tensorflow/tensorboard/tensorboard.py --logdir=${PWD}
```
To launch the prediction app:

Run
```
python number_prediction_app.py
```
Open Browser at localhost:8080


### About this Project
This repo is an app that uses a Convolutional Neural Network built in Tensorflow and trained on MNIST data to predict a number drawn by a user in d3.js
