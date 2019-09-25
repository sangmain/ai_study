import tensorflow as tf

x = [1,2,3]
y = [1,2,3]


w = tf.Variable(5.0)

hypothesis = x * w
cost = tf.reduce_mean(tf.square(hypothesis - y))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(101):
        _, w_val = sess.run([train, w])
        print(step, w_val)
