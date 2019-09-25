import tensorflow as tf
import numpy as np
np_load_old = np.load
tf.set_random_seed(666)
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

xy = np.load("./data/boston.npy")

x_data = xy[0][0]
y_data = xy[0][1]

x_test = xy[1][0]
y_test = xy[1][1]

print(x_data.shape)
print(x_test.shape)

tf.set_random_seed(777)

y_data = y_data.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
# print(x_data.shape, y_data.shape)
# x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], 1)

x = tf.placeholder(tf.float32, [None, 13])
y = tf.placeholder(tf.float32, [None, 1])

L1 = tf.layers.dense(x, 8, activation=tf.nn.leaky_relu)
# L2 = tf.layers.dense(L1, 20, activation=tf.nn.relu)
L3 = tf.layers.dense(L1, 1, activation=tf.nn.leaky_relu)

hypothesis = L3

cost = tf.reduce_mean(tf.square(tf.subtract(hypothesis, y)))

train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

session = tf.Session()

# Initializes global variables in the graph.
session.run(tf.global_variables_initializer())

for step in range(10001):
    cost_val, hy_val, _ = session.run([cost, hypothesis, train], feed_dict = {x : x_data, y : y_data})
    
    
    if step % 500 == 0:
        print(step, "Cost : ", cost_val)

predict = session.run([hypothesis], feed_dict = {x : x_test})

predict = np.array(predict)
y_test_reshape = y_test.reshape((-1, ))
predict = predict.reshape((-1, ))

from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test_reshape, predict)
print("R2: ", r2_y_predict)

import random
r = random.randint(0, x_test.shape[0] - 1)
print("Label: ", session.run(tf.argmax(y_test[r:r + 1], 1)))
print("Prediction: ", session.run(tf.argmax(L3, 1), feed_dict = {x: x_test[r:r + 1]}))  