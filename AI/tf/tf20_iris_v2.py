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


L1 = tf.layers.dense(x, 100, activation=tf.nn.relu)
L2 = tf.layers.dense(L1, 20, activation=tf.nn.relu)
L3 = tf.layers.dense(L2, 3, activation=tf.nn.softmax)


# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=L3, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# predicted = tf.cast(hypothesis > 0.1, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(L3, y), dtype=tf.float32))
print(x_test)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val = sess.run([optimizer, cost], feed_dict={x: x_train, y:y_train})

        if step % 200 == 0:
            print(step, cost_val)

    correct_prediction = tf.equal(tf.argmax(L3, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', sess.run(accuracy, feed_dict={x: x_test, y: y_test}))
    
    r = random.randint(0, x_test.shape[0] - 1)
    print("Label: ", sess.run(tf.argmax(y_test[r:r + 1], 1)))
    print("Prediction: ", sess.run(tf.argmax(L3, 1), feed_dict = {x: x_test[r:r + 1]}))