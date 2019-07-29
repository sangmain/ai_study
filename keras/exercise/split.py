import numpy as np

x = np.array(range(1,101))
y = np.array(range(1,101))


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size= 0.4)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, random_state=66, test_size=0.5)

print(x_train)
print(x_val)
print(x_test)