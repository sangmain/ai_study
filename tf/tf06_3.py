import tensorflow as tf
tf.set_random_seed(777)

x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

w = tf.Variable([0.3], tf.float32)
b = tf.Variable([-0.3], tf.float32)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

hypothesis = x * w + b

#cost loss function
cost = tf.reduce_sum(tf.square(hypothesis - y))

# optimzier
train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)
 
#train = optimizer.minimize(cost)

# Launch the graph in a session
with tf.Session() as sess:
    #Initializes global variables in the graph
    sess.run(tf.global_variables_initializer())

    #Fit the Line
    for step in range(1000):
        sess.run(train, {x:x_train, y:y_train})

    #evaluate train accuracy
    w_val, b_val, cost_val = sess.run([w, b, cost], feed_dict={x:x_train, y:y_train})
    print(f"w: {w_val} b: {b_val} cost: {cost_val}")