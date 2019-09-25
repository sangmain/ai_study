import numpy as np
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

xy = np.load("./data/iris.npy", )

x = xy[:, :-1]
y = xy[:, [-1]]

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

cd = LabelEncoder()
cd.fit(y)
y = cd.transform(y)
y = to_categorical(y)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 66, test_size = 0.2)

print(x_train.shape)
print(x_test.shape)


import tensorflow as tf
import random

tf.set_random_seed(777)

x = tf.placeholder("float", [None, 4])
y = tf.placeholder("float", [None, 3])


w = tf.get_variable('w1', shape=[4, 100], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([100]))
layer1 = tf.nn.relu(tf.matmul(x, w) + b)

w = tf.get_variable('w2',shape=[100, 20], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([20]))
layer2 = tf.nn.relu(tf.matmul(layer1, w) + b)


w = tf.get_variable('w3',shape=[20, 3], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([3]))
hypothesis = tf.nn.softmax(tf.matmul(layer2, w) +  b)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))
print(x_test)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val = sess.run([optimizer, cost], feed_dict={x: x_train, y:y_train})

        if step % 200 == 0:
            print(step, cost_val)

    print('------------')
    predicted = sess.run(hypothesis, feed_dict = {x:x_test})
    print(predicted, sess.run(tf.argmax(predicted, 1)))

    acc = accuracy.eval(feed_dict={x:x_test, y:y_test})
    print("acuaracy:", acc)