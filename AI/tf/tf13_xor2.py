import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)



x = tf.placeholder(tf.float32, shape=[None,2])
y = tf.placeholder(tf.float32, shape=[None,1])

w = tf.Variable(tf.random_normal([2,100]), name='weight')
b = tf.Variable(tf.random_normal([100]), name='bias')
layer1 = tf.sigmoid(tf.matmul(x, w) + b)


w = tf.Variable(tf.random_normal([100,100000]), name='weight')
b = tf.Variable(tf.random_normal([100000]), name='bias')
layer2 = tf.sigmoid(tf.matmul(layer1, w) + b)

w = tf.Variable(tf.random_normal([100000,20]), name='weight')
b = tf.Variable(tf.random_normal([20]), name='bias')
layer3 = tf.sigmoid(tf.matmul(layer2, w) + b)


w = tf.Variable(tf.random_normal([20,12]), name='weight')
b = tf.Variable(tf.random_normal([12]), name='bias')
layer4 = tf.sigmoid(tf.matmul(layer3, w) + b)

w = tf.Variable(tf.random_normal([12,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
hypothesis = tf.sigmoid(tf.matmul(layer4, w) + b)

cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))

train = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

#x, y, w, b, hypothesis, cost, train
#sigmoid
#predict accuracy

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        _, cost_val, w_val = sess.run(
            [train, cost, w], feed_dict={x:x_data, y:y_data}
        )

        if step % 100 == 0:
            print(step, cost_val, w_val)

    h, c, a = sess.run(
        [hypothesis, predicted, accuracy], feed_dict={x: x_data, y: y_data}
    )
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)