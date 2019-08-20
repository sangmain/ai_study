#랜덤값으로 변수 1개를 만들고 변수의 내용 출력
import tensorflow as tf

w = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

sess = tf.Session()
sess.run((tf.global_variables_initializer())
aaa = w.eval(session=sess))

print(sess.run(w))