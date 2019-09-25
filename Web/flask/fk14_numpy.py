import pymssql as ms
import numpy as np

conn = ms.connect(server='localhost', user='bitcamp', password='1234', database='BTDB')

cursor = conn.cursor()

cursor.execute('SELECT TOP(1000) * FROM train;')

row = cursor.fetchall()
print(row)
conn.close()

aaa = np.asarray(row)
print(aaa)
print(aaa.shape)
print(type(aaa))

np.save('test_aaa.npy', aaa)