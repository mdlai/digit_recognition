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


### Fixes for new versions of TF
1)
Error --> ModuleNotFoundError: No module named 'tensorflow.tensorboard.tensorboard'
Solution --> https://github.com/tensorflow/tensorflow/issues/10959

Run
from: `python ~/path/to/python/tensorflow/tensorboard/tensorboard.py --logdir=${PWD}`
to: `tensorboard --logdir=output`

2)
Error --> 'module' object has no attribute 'scalar_summary'
Solution --> https://stackoverflow.com/questions/41066244/tensorflow-module-object-has-no-attribute-scalar-summary

Change code
from: `tf.scalar_summary(...)`
to: `tf.summary.scalar(...)`

3)
Error --> AttributeError: 'module' object has no attribute 'merge_all_summaries'
Solution --> https://github.com/tensorflow/models/issues/1066

Change code
from: `summary_op = tf.merge_all_summaries()`
to: `summary_op = tf.summary.merge_all()`
