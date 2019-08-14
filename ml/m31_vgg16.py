from keras.applications import VGG19, VGG16
from keras.applications import 
, InceptionV3, ResNet50, MobileNet 

from keras.layers import Dense
from keras.models import Sequential
# conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
# conv_base = VGG16()

a = VGG16()
b = Xception()
c = InceptionV3()
d = ResNet50()
e = MobileNet()


# conv_base.summary()
# model = Sequential(conv_base.layers)
# model.add(Dense(256, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.summary()

# model = Sequential()
# model.add(conv_base)
# model.add(Dense(256, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.summary()