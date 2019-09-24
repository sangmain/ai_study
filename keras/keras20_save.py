#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout

model = Sequential()

from keras import regularizers
model.add(Dense(10, input_shape = (3, ), activation ='relu')
                #kernel_regularizer= regularizers.l1(0.0001)))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(25))
model.add(Dense(8))
model.add(Dense(1))



# model.add(Dense(10, input_shape = (3, ), activation ='relu',
#                 kernel_regularizer= regularizers.l1(0.01)))
# model.add(Dense(30))
# model.add(Dense(40))
# model.add(Dense(25))
# model.add(Dense(8))
# model.add(Dense(1))
#model.summary()

model.save('savetest01.h5')
print('저장 완료')