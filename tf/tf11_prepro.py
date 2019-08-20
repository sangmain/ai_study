import numpy as np
import tensorflow as tf

xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973], 
                [823.02002, 828.070007, 1828100, 821.655029, 828.070007], 
                [819.929993, 824.400024, 1438100, 818.97998, 824.159973], 
                [816, 820.958984, 1008100, 815.48999, 819.23999], 
                [819.359985, 823, 1188100, 818.469971, 818.97998], 
                [819, 823, 1198100, 816, 820.450012], 
                [811.700012, 815.25, 1098100, 809.780029, 813.669983], 
                [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])

x_data = xy[:, :-1]
y_data = xy[:, [-1]]

print(x_data.shape)
print(y_data.shape)


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score

scaler = StandardScaler()
# scaler = MinMaxScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)


x = tf.placeholder('float', [None, 4])
y = tf.placeholder('float', [None, 1])

w = tf.Variable(tf.random_normal([4,1]))
b = tf.Variable(tf.random_normal([1]))

hypothesis = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(hypothesis - y))

train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(101):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={x:x_data, y:y_data})
    print(step, "Cost:", cost_val, "\nPrediction:\n", hy_val)

r2_y_predict = r2_score(y_data, hy_val)
print("R2: ", r2_y_predict)