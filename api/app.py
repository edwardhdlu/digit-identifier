import tensorflow as tf
import numpy as np
import os
from flask import Flask, request

app = Flask(__name__)

with app.app_context():
	x = tf.placeholder(tf.float32, shape=[None, 1024])

	# convolution
	def conv2d(x, W):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

	# max pooling
	def max_pool_2x2(x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

	# first layer
	W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
	b_conv1 = tf.Variable(tf.zeros([32]))

	x_image = tf.reshape(x, [-1, 32, 32, 1])

	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	# second layer
	W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
	b_conv2 = tf.Variable(tf.zeros([64]))

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	# third layer
	W_fc1 = tf.Variable(tf.truncated_normal([8 * 8 * 64, 1024], stddev=0.1))
	b_fc1 = tf.Variable(tf.zeros([1024]))

	h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	# dropout
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	# fourth layer
	W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
	b_fc2 = tf.Variable(tf.zeros([10]))

	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	# restore saved data
	session = tf.Session()
	saver = tf.train.Saver()
	saver.restore(session, "static/model")

@app.route("/")
def home():
	return "This server returns SHVN model predictions."

@app.route("/predict/", methods = ["GET", "POST"])
def predict():
	size = 32
	query = request.data

	if len(query) == 0:
		return "Nothing here!"

	arr = [float(i) for i in query.split(",")]

	if len(arr) != size * size:
		return "Invalid input!"

	X = np.array([arr]).astype(np.float32)

	prediction = session.run(y_conv, feed_dict={x: X, keep_prob: 1.0})
	return np.array_str(prediction)

if __name__ == "__main__":
 	app.debug = True
 	port = int(os.environ.get("PORT", 5000))
 	app.run(host="0.0.0.0", port=port)