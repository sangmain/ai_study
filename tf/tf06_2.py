import tensorflow as tf
tf.set_random_seed(777)

w = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

hypothesis = x * w + b

#cost loss function
cost = tf.reduce_mean(tf.square(hypothesis - y))

# optimzier
train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)
 
# Launch the graph in a session
with tf.Session() as sess:
    #Initializes global variables in the graph
    sess.run(tf.global_variables_initializer())

    #Fit the Line
    for step in range(2001):
        _, cost_val, w_val, b_val = sess.run( ,
         feed_dict={x:[1,2,3,4,5], y:[2.1, 3.1, 4.1, 5.1, 6.1]})

        if step % 20 == 0:
            print(step, cost_val, w_val, b_val )

    #Testing our model #predict
    print(sess.run(hypothesis, feed_dict={x:[5]}))
    print(sess.run(hypothesis, feed_dict={x:[2.5]}))
    print(sess.run(hypothesis, feed_dict={x:[1.5, 3.5]}))