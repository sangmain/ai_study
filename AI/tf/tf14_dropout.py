import tensorflow as tf
import matplotlib.pyplot as plt
import random

tf.set_random_seed(777)

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

## 코딩하시오 x, y, w, b, hypothesis, cosst train
x = tf.placeholder("float", [None, 28*28])
y = tf.placeholder("float", [None, 10])
nb_classes = 10

w = tf.Variable(tf.random_normal([28*28, 32]), name='weight')
b = tf.Variable(tf.random_normal([32]), name='bias')
layer1 = tf.nn.sigmoid(tf.matmul(x, w) +  b)

layer1 = tf.nn.dropout(layer1, keep_prob = 0.5)

w = tf.Variable(tf.random_normal([32, 64]), name='weight')
b = tf.Variable(tf.random_normal([64]), name='bias')
layer2 = tf.nn.sigmoid(tf.matmul(layer1, w) +  b)

layer2 = tf.nn.dropout(layer2, keep_prob = 0.5)

# w = tf.Variable(tf.random_normal([64, 128]), name='weight')
# b = tf.Variable(tf.random_normal([128]), name='bias')
# layer3 = tf.nn.sigmoid(tf.matmul(layer2, w) +  b)

# layer3 = tf.nn.dropout(layer3, keep_prob = 0.5)

# w = tf.Variable(tf.random_normal([128, 256]), name='weight')
# b = tf.Variable(tf.random_normal([256]), name='bias')
# layer4 = tf.nn.sigmoid(tf.matmul(layer3, w) +  b)

# layer4 = tf.nn.dropout(layer4, keep_prob = 0.5)

# w = tf.Variable(tf.random_normal([256, 512]), name='weight')
# b = tf.Variable(tf.random_normal([512]), name='bias')
# layer5 = tf.nn.sigmoid(tf.matmul(layer4, w) +  b)

# layer5 = tf.nn.dropout(layer5, keep_prob = 0.5)


w = tf.Variable(tf.random_normal([64, 30]), name='weight')
b = tf.Variable(tf.random_normal([30]), name='bias')
layer6 = tf.nn.sigmoid(tf.matmul(layer2, w) +  b)

layer6 = tf.nn.dropout(layer6, keep_prob = 0.5)

w = tf.Variable(tf.random_normal([30, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')
hypothesis = tf.nn.softmax(tf.matmul(layer6, w) +  b)



cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))

train = tf.train.AdamOptimizer(learning_rate=0.00027, beta1=0.9, beta2=0.999).minimize(cost)
#.0001 = 71
#.00005 = 51
#.0002 = 82
#.0003 = 0.098
#.00025 = 0.8434
#.00027 = 0.8485
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

num_epochs = 15
batch_size = 100
num_iterations = int(mnist.train.num_examples / batch_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(num_epochs):
        avg_cost = 0

        for i in range(num_iterations):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, cost_val = sess.run([train, cost], feed_dict = {x: batch_xs, y:batch_ys})
            avg_cost += cost_val / num_iterations

        print("Epoch: {:04d}, Cost: {:.9f}".format(epoch + 1, avg_cost))

    print("Learning finished")

    print(
        "Accuracy: ",
        accuracy.eval(
            session=sess, feed_dict={x:mnist.test.images, y:mnist.test.labels}
        ),
    )

    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r : r + 1], 1)))
    print(
        "Prediction: ",
        sess.run(tf.argmax(hypothesis, 1), feed_dict={x: mnist.test.images[r : r + 1]}),
    )

    plt.imshow(
        mnist.test.images[r : r + 1].reshape(28, 28),
        cmap='Greys',
        interpolation='nearest',
    )
    plt.show()