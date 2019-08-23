import tensorflow as tf
import random

from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(777)
mnist = input_data.read_data_sets("MNIST_data", one_hot = True)

#hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1]) #img 28 x 28 x 1
Y = tf.placeholder(tf.float32, [None, 10])
W1 = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01)) #커널 사이즈 3,3 채널 1, 출력 노드 32
print('W1: ', W1)
L1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides = [1,2,2,1], padding='SAME')



W2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides = [1,1,1,1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob = 0.25)
print("L1: ", L2)

W3 = tf.Variable(tf.random_normal([3,3, 64, 128], stddev=0.01))
L3 = tf.nn.conv2d(L2, W3, strides = [1,1,1,1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides = [1,2,2,1], padding='SAME')
L3_flat = tf.reshape(L3, [-1, 4 * 4 * 128])

W4 = tf.Variable(tf.random_normal([3,3, 128, 256], stddev=0.01))
L4 = tf.nn.conv2d(L3, W4, strides = [1,1,1,1], padding='SAME')
L4 = tf.nn.relu(L4)
L4 = tf.nn.max_pool(L4, ksize=[1,2,2,1], strides = [1,2,2,1], padding='SAME')
L4_flat = tf.reshape(L4, [-1, 2 * 2 * 256])

W5= tf.get_variable("W5", shape =[2 * 2 * 256, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L4_flat, W5) + b
# logits = tf.nn.dropout(logits1, keep_prob = 0.25)

# W5= tf.get_variable("W5", shape =[2 * 2 * 256, 512], initializer=tf.contrib.layers.xavier_initializer())
# b = tf.Variable(tf.random_normal([512]))
# logits1 = tf.matmul(L4_flat, W5) + b
# logits1 = tf.nn.dropout(logits1, keep_prob = 0.25)

# W6= tf.get_variable("W6", shape =[512, 30], initializer=tf.contrib.layers.xavier_initializer())
# b = tf.Variable(tf.random_normal([30]))
# logits2 = tf.matmul(logits1, W6) + b

# W7= tf.get_variable("W7", shape =[30, 10], initializer=tf.contrib.layers.xavier_initializer())
# b = tf.Variable(tf.random_normal([10]))
# logits = tf.matmul(logits2, W7) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

print("Learning started. It takes some time")
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict = feed_dict)
        avg_cost += c / total_batch

    print("Epoch: ", "%04d" % (epoch + 1), "cost = ", "{:.9f}".format(avg_cost))

print("Learning Finished")

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Accuracy:", sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))


r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r + 1]}))