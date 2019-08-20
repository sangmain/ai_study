import tensorflow as tf
tf.set_random_seed(777)
import numpy as np
import pandas as pd

# a = pd.read_csv('')
data = np.genfromtxt("./data/data-04-zoo.csv", delimiter=',')
x_data = data[:-1, 0:-1]
y_data = data[:-1, [-1]]

print(x_data.shape, y_data.shape)
x_test = data[-10:, 0:-1]
y_test = data[-10:, [-1]]


print(x_test.shape, y_test.shape)

from keras.utils import to_categorical
y_data = to_categorical(y_data)
y_test = to_categorical(y_test)
# print(x_data.shape, y_data.shape)

x = tf.placeholder("float", [None, 16])
y = tf.placeholder("float", [None, 7])
nb_classes = 7

w = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(x, w) +  b)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val = sess.run([optimizer, cost], feed_dict={x: x_data, y:y_data})

        if step % 200 == 0:
            print(step, cost_val)

    print('------------')
    predicted = sess.run(hypothesis, feed_dict = {x:x_test})
    print(predicted, sess.run(tf.argmax(predicted, 1)))

    acc = accuracy.eval(feed_dict={x:x_test, y:y_test})
    print("acuaracy:", acc)