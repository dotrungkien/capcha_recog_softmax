from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1
model_path = "tmp/model.ckpt"

d = {'C': [1,0,0,0,0,0,0,0],
    'E': [0,1,0,0,0,0,0,0],
    'F': [0,0,1,0,0,0,0,0],
    'M': [0,0,0,1,0,0,0,0],
    'N': [0,0,0,0,1,0,0,0],
    'P': [0,0,0,0,0,1,0,0],
    'T': [0,0,0,0,0,0,1,0],
    'W': [0,0,0,0,0,0,0,1],}
chars = {0:'C', 1:'E', 2: 'F', 3:'M', 4:'N', 5: 'P', 6:'T', 7:'W'}


def img_preprocessing():
    images = []
    labels = []
    all_files = [f for f in listdir('dataset/samples/') if isfile(join('dataset/samples/', f))]
    for name in all_files:
        if 'png' in name:
            labels.append(d[name.split('-')[0]])
            img_path = 'dataset/samples/' + name
            img = np.array(Image.open(img_path).convert('L')).reshape(1024,)
            for i in range(1024):
                img[i] = img[i]/255.
            images.append(img)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

images, labels = img_preprocessing()
n = len(images)
n_train = int(n*0.8)
images_train = images[:n_train]
labels_train = labels[:n_train]
images_test = images[n_train:]
labels_test = labels[n_train:]

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 1024])
y = tf.placeholder(tf.float32, [None, 8])

# Set model weights
W = tf.Variable(tf.zeros([1024, 8]))
b = tf.Variable(tf.zeros([8]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()
def train():
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = 1000
            # Loop over all batches
            for i in range(total_batch):
                _, c = sess.run([optimizer, cost], feed_dict={x: images_train,
                                                              y: labels_train})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if (epoch+1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", accuracy.eval({x: images_test, y: labels_test}))

        #save model
        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)

def predict(img_path):
    img = np.array(Image.open(img_path).convert('L')).reshape(1024,)
    for i in range(1024): img[i] = img[i]/255.
    # print("Starting 2nd session...")
    with tf.Session() as sess:
        # Initialize variables
        sess.run(init)

        # Restore model weights from previously saved model
        saver.restore(sess, model_path)
        predict = tf.argmax(pred, 1)
        # print("Predict = {0}, actual = {1}".format(predict.eval({x:[img]}), img_path))
        return chars[predict.eval({x:[img]})[0]]
# train()
# print(chars[predict('M-rdrfn.png')])