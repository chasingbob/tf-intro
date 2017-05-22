from __future__ import division, print_function, unicode_literals

import csv
import tensorflow as tf
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

data = pd.read_csv('iris.data')

# To get a feel for the data
data.head()

data_X = np.array(data.ix[:,:-1])
data_y = np.array(data.ix[:,-1:])

print(data_X[:5])
print(data_y[:5])

le = LabelEncoder()
le.fit(data_y)
print(le.classes_)
data_y = le.transform(data_y)
print(data_y)

# Train/Test split
train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=0.15, random_state=42)
print(train_X)
print(train_y)


def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 1 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="weights")
        b = tf.Variable(tf.zeros([n_neurons]), name="biases")
        Z = tf.matmul(X, W) + b
        if activation=="relu":
            return tf.nn.relu(Z)
        else:
            return Z

n_inputs = 4  
n_hidden1 = 50
n_outputs = 3
display_step = 10
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

with tf.name_scope("nn"):
    hidden1 = neuron_layer(X, n_hidden1, "hidden1", activation="relu")
    logits = neuron_layer(hidden1, n_outputs, "output")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    predict = tf.argmax(logits,1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 60

# Training
with tf.Session() as sess:
    init.run()
    print('Training started...')
    for epoch in range(n_epochs):
        sess.run(training_op, feed_dict={X: train_X, y: train_y})
        acc_train = accuracy.eval(feed_dict={X: train_X, y: train_y})
        acc_test = accuracy.eval(feed_dict={X: test_X, y: test_y})
        if epoch % display_step == 0:
            print('Epoch: {} Train acc: {} Test acc: {}'.format(epoch, acc_train, acc_test))

    save_path = saver.save(sess, "./model.ckpt")


    print('\r\n A few examples: ')
    for i in range(len(test_X)):
        pred = predict.eval(feed_dict={X: test_X[i:i+1], y: test_y[i:i+1]})
        pred_ = le.inverse_transform(pred)
        y_ = le.inverse_transform(test_y[i:i+1])

        print('Values: {} Predict: {} {} Actual: {} {}'.format(test_X[i:i+1],pred[0], pred_, test_y[i], y_))

