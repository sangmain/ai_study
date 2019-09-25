import numpy as np
import pandas as pd

dataset = pd.read_csv("./data/test0822.csv", delimiter=",")

cnt = 0
index = []
for date in dataset['date'].values:

    if "-07-11" in date:
        print(date)
        index.append(cnt)
    cnt += 1

print(index)

row_per_col = 50
result = []
for i in index:
    data = dataset[i-row_per_col:i+5].values
    result.append(data)


result = np.array(result)

x_train = result[:, :-5]
y_train = result[:, [-5]]
print(x_train[0])
print(y_train[0])