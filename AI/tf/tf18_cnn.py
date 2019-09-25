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
L2 = tf.nn.conv2d(L1, W2, strides = [1,1,1,1], padding='same')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print("L1: ", L2)
L2_flat = tf.reshape(L2, [-1, 7 * 7 * 64])