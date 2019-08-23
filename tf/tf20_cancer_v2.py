import tensorflow as tf
import numpy as np
import random
data = np.load("./data/cancer.npy")
x_data = data[:, 0:-1]
y_data = data[:, [-1]]

tf.set_random_seed(777)

y_data = y_data.reshape(-1, 1)
print(x_data.shape, y_data.shape)

x = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None, 1])

L1 = tf.layers.dense(x, 1, activation=tf.nn.sigmoid)

logits = L1

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits = logits, labels = y )
    )

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

predicted = tf.cast(logits > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, logits], feed_dict={x:x_data, y:y_data})
        if step % 200 == 0:
            print(step, cost_val)

    a = sess.run([accuracy], feed_dict = {x: x_data, y: y_data})
    print("Accuracy: ", a)
    r = random.randint(0, x_data.shape[0] - 1)
    print("Label: ", y_data[r:r+1])
    print("Prediction: ", sess.run(tf.argmax(logits, 1), feed_dict = {x: x_data[r:r + 1]}))