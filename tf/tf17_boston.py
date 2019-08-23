import tensorflow as tf
import numpy as np
np_load_old = np.load

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

w = tf.get_variable(name = 'w1', shape = [13, 8], initializer = tf.zeros_initializer())
b = tf.Variable(tf.random_normal([8]))
layer1 = tf.nn.leaky_relu(tf.matmul(x, w) + b)

w = tf.get_variable(name = 'w2', shape = [8, 1], initializer = tf.zeros_initializer())
b = tf.Variable(tf.random_normal([1]))
hypothesis = tf.nn.leaky_relu(tf.matmul(layer1, w) + b)

cost = tf.reduce_mean(tf.square(tf.subtract(hypothesis, y)))

train = tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(cost)

session = tf.Session()

# Initializes global variables in the graph.
session.run(tf.global_variables_initializer())

for step in range(10001):
    cost_val, hy_val, _ = session.run([cost, hypothesis, train], feed_dict = {x : x_data, y : y_data})
    print(step, "Cost : ", cost_val)

predict = session.run([hypothesis], feed_dict = {x : x_test})

predict = np.array(predict)
y_test_reshape = y_test.reshape((-1, ))
predict = predict.reshape((-1, ))

from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test_reshape, predict)
print("R2: ", r2_y_predict)