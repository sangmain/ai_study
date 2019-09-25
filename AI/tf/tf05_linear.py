import tensorflow as tf
tf.set_random_seed(777)

# X and Y data
x_train = [1,2,3]
y_train = [1,2,3]

w = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x_train * w + b

# cost / loss function
loss = tf.reduce_mean(tf.square(hypothesis - y_train))

# optimizer
train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(loss)

#Launch the graph in a session.
with tf.Session() as sess:
    
    #Initializes global variables in the graph
    sess.run(tf.global_variables_initializer()) #★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

    #Fit the line
    for step in range(2001):
        _, cost_val, w_val, b_val = sess.run([train, loss, w, b])

        if step % 20 == 0:
            print(step, cost_val, w_val, b_val)

        