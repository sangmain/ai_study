import tensorflow as tf
import numpy as np

xy = np.load("./data/cancer.npy")
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

tf.set_random_seed(777)

y_data = y_data.reshape(-1, 1)
print(x_data.shape, y_data.shape)

x = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.get_variable('w1', shape=[30, 100], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([100]))
layer1 = tf.nn.relu(tf.matmul(x, w) + b)

w = tf.get_variable('w2',shape=[100, 20], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([20]))
layer2 = tf.nn.relu(tf.matmul(layer1, w) + b)


w = tf.get_variable('w3',shape=[20, 1], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([1]))
logits = tf.matmul(layer2, w) + b
hypothesis = tf.nn.sigmoid(logits)

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits = logits, labels = y )
    )

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={x:x_data, y:y_data})
        if step % 200 == 0:
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, predicted, accuracy],
                        feed_dict = {x: x_data, y: y_data})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c , "\nAccuracy: ", a)