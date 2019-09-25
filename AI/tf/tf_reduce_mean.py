import tensorflow as tf

x = tf.constant([[1., 3.], [2., 6.]])

sess = tf.Session()

# print(sess.run(x))
# print(sess.run(tf.reduce_mean(x)))
# print(sess.run(tf.reduce_mean(x, 0))) # 행 끼리 평균
# print(sess.run(tf.reduce_mean(x, 1))) # 열 끼리 평균

x1 = tf.constant([[6., 8.], [1., 4.]])

multiplied = tf.matmul(x, x1)
print(sess.run(x))
print()
print(sess.run(x1))
print()
print(sess.run(multiplied))