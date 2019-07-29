import numpy as np

x1 = np.array([range(100), range(311, 411), range(100)])
y1 = np.array([range(501, 601), range(711, 811), range(100)])

x2 = np.array([range(100, 200), range(311, 411), range(100, 200)])
y2 = np.array([range(501, 601), range(711, 811), range(100)])


print(x1.shape)
x1 = np.transpose(x1)
print(x1.shape)


from keras.models import Model, Sequential
from keras.layers import Input, Dense

input1 = Input(shape=(3,))
dense1 = Dense(100, activation = 'relu')(input1)
dense2 = Dense(30)(dense1)
dense3 = Dense(2)(dense2)

input2 = Input(shape=(3,))
dense4 = Dense(40, activation='relu')(input2)

from keras.layers.merge import concatenate
merge1 = concatenate([dense3, dense4])

output1 = Dense(10)(merge1)
output2 = Dense(50)(merge1)

model = Model(inputs=[input1, input2], outputs=[output1,output2])

model.summary()