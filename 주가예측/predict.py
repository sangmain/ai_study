import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt

train = pd.read_csv("cheatkey.csv")

end_prices = train['종가']

#normalize window
seq_len = 50

result = []
result[:] = end_prices[-seq_len:]
print(len(result))



#normalize data
def normalize_windows(data):
    normalized_data = []
    for x in data:
        normalized_x = [((float(x) / float(data[0])) - 1)]
        normalized_data.append(normalized_x)
    return np.array(normalized_data)

x_test = normalize_windows(result)

x_test = x_test.reshape(1, -1, 1)

print(x_test.shape)
from keras.models import load_model

model = load_model("model.h5")

pred = model.predict(x_test)

index = len(x_test)
pred = model.predict(x_test)
print('역정규화 개시')

x_today = []
un_norm = result[0]
for x in x_test[index-1]:
    x_today.append((x+1) * un_norm)

pred_today = (pred[index-1]+1) * un_norm

x_today = np.array(x_today)
print("last 5 days:\n ", x_today)
print("prediction: ", pred_today)