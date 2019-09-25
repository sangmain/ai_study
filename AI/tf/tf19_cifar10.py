import tensorflow as tf
from keras.datasets import cifar10
from keras.utils import to_categorical
import numpy as np

def next_batch(number, data, labels):
    '''
    `num` 개수 만큼의 랜덤한 샘플들과 레이블들을 리턴합니다.
    '''
    index = np.arange(0 , len(data))
    np.random.shuffle(index)
    index = index[ : number]
    data_shuffle = [data[i] for i in index]
    labels_shuffle = [labels[i] for i in index]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

    
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)
print(y_train.shape)

learning_rate = 0.001
training_epochs = 15
batch_size = 32

X = tf.placeholder(tf.float32, [None, 32,32,3])
# X_img = tf.reshape(X, [-1, 32, 32, 3])
Y = tf.placeholder(tf.int32, [None, 1])

Y_one_hot = tf.one_hot(Y, 10)
Y_one_hot = tf.reshape(Y_one_hot, [-1, 10])

W = tf.Variable(tf.random_normal([3,3,3,32], stddev=0.01)) #커널 사이즈 3,3 채널 1, 출력 노드 32
L1 = tf.nn.conv2d(X, W, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

L1_flat = tf.reshape(L1, [-1, 16 * 16 * 32])

W = tf.get_variable("W2", shape = [ 16 * 16 * 32, 512], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([512]))
L2 = tf.matmul(L1_flat, W) + b
L2 = tf.nn.relu(L2)

W = tf.get_variable("W3", shape = [512, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
L3 = tf.matmul(L2, W) + b
logits = tf.nn.softmax(L3)



cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

# initialize
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    training_epochs = 5
    
    # train my model, Model Fit
    acc = 0
    print('Learning started. It takes sometime.')
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = x_train.shape[0] // batch_size
        for i in range(total_batch):
            batch_xs, batch_ys = next_batch(batch_size, x_train, y_train)
            feed_dict = {X: batch_xs, Y: batch_ys}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))

    print('Learning Finished!')

    # Test model and check accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy : ', sess.run(accuracy, feed_dict = {X : x_test, Y : y_test}))

    import random
    # Get one and predict
    r = random.randint(0, 10000 - 1)
    print("Label : ", sess.run(tf.argmax(y_test[r : r + 1], 1)))
    print("Prediction : ", sess.run(tf.argmax(logits, 1), feed_dict = {X : x_test[r : r + 1]}))