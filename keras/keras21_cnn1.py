from keras.models import Sequential

filter_size = 32
kernel_size = (3,3)


from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout
model = Sequential()


model.add(Conv2D(3, (3,3), input_shape=(6,6,1)))

# # model.add(Conv2D(7, (2,2), padding='same', input_shape=(5,5,1)))

# model.add(Conv2D(7, (2,2), padding='same', input_shape=(10,10,1)))
# # model.add(Conv2D(16,(2,2)))
# model.add(MaxPooling2D(3,3))
# # model.add(Conv2D(8,(2,2)))
# model.add(Dropout(0.2))
# ## CNN의 마지막
# # model.add(Flatten())
# # model.add(Dense(30))

model.summary()