import flask
import tensorflow as tf
import numpy as np

#---------- MODEL IN MEMORY ----------------#

# Load the model from tensorflow,
# Use it to predict

#Global Variable
global current_image
current_image = np.zeros((1,784))

### Helper Functions
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='weights')

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='biases')

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name='convolution')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME', name='pool')

### Graph of CNN
with tf.Graph().as_default():
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y_ = tf.placeholder(tf.float32, [None, 10])

#Inference
    with tf.name_scope('hidden1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        x_image = tf.reshape(x, [-1,28,28,1])
        hidden1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    with tf.name_scope('hidden2'):
        h_pool1 = max_pool_2x2(hidden1)
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('fully_connected'):
        h_pool2 = max_pool_2x2(h_conv2)
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        fully_connected = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        dropout = tf.nn.dropout(fully_connected, keep_prob)

    with tf.name_scope('softmax'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        y_conv=tf.nn.softmax(tf.matmul(dropout, W_fc2) + b_fc2)

#Loss
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv), name='xentropy')

#Training
    tf.scalar_summary(cross_entropy.op.name, cross_entropy)
    global_step=tf.Variable(0,name='global_step',trainable=False)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy,global_step=global_step)

#Evaluation
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Log Creation
    summary_op = tf.merge_all_summaries()
    sess = tf.Session()
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

#Restore
    saver.restore(sess,"../data/model.ckpt")

#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = flask.Flask(__name__)

# Homepage
@app.route("/")
def viz_page():
    with open("test.html", 'r') as viz_file:
        return viz_file.read()

# Get an example and return it's score from the predictor model
@app.route("/score", methods=["POST"])
def score():
    # Get decision score for our example that came with the request
    data = flask.request.json

#Alternate Method?
    # if data["example"][0] == [-100]:
    #     current_image = np.zeroes((1,768))
    # for i in data["example"]:
    #     if i >= 0 and i < 784:
    #         print "made it"
    #         # current_image[0][i] = 1
    #     else:
    #         pass
    #     print "made it?"
    # print current_image
    # y_fit = sess.run(tf.argmax(y_conv,1),feed_dict={x: current_image, keep_prob:1})
    # results = {"score": y_fit[0]}
    # return flask.jsonify(results)

    current_image = np.zeros((1,784))
    for i in data["example"]:
        if i >= 0 and i < 784:
            current_image[0][i] = 1
    y_fit = sess.run(tf.argmax(y_conv,1),feed_dict={x: current_image, keep_prob:1})
    results = {"score": y_fit[0]}
    return flask.jsonify(results)

#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 8080
# (The default website port)
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)
