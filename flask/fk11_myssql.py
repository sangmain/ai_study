import pymssql as ms

conn = ms.connect(server='localhost', user='bitcamp', password='1234', database='BTDB')

cursor = conn.cursor()

cursor.execute('SELECT TOP(1000) * FROM train;')

row = cursor.fetchone()
print(type(row))        ##  tuple

while row:
    # print("첫컬럼=%s, 둘컬럼=%s" %(row[0], row[1]))
    print(row)
    row = cursor.fetchone()

conn.close()