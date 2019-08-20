import tensorflow as tf
import numpy as np
tf.set_random_seed(777)


xy = np.loadtxt('./data/data-04-zoo.csv', delimiter=',', dtype=np.float32)

x_data = xy[:-1, 0:-1]
y_data = xy[:-1, [-1]]

print(x_data.shape, y_data.shape)
x_test = xy[-10:, 0:-1]
y_test = xy[-10:, [-1]]

nb_classes = 7
x = tf.placeholder(tf.float32, [None, 16])
y = tf.placeholder(tf.int32, [None, 1])

y_one_hot = tf.one_hot(y, nb_classes)
print("one_hot", y_one_hot)

y_one_hot = tf.reshape(y_one_hot, [-1, nb_classes])
print('reshape one_hot: ', y_one_hot)

w = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(x, w) + b
hypothesis = tf.nn.softmax(logits)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits = logits, labels=tf.stop_gradient([y_one_hot]))
    )
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(y_one_hot, 1))
accuracy = tf.reduce_mean(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val, acc_val = sess.run([optimizer, cost, accuracy], feed_dict={x: x_data, y:y_data})

        if step % 100 == 0:
            print("step: {:5}\t Cost: {:.3f}\t Acc: {:.2%}".format(step, cost_val, acc_val))

    pred = sess.run(prediction, feed_dict = {x: x_data})

    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
