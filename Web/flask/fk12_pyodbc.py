import pyodbc as pyo


server = 'localhost'
database = 'BTDB'
username = 'bitcamp'
password = '1234'

cnxn = pyo.connect('DRIVER={ODBC Driver 13 for SQL Server}; SERVER=' +server+
                    '; PORT=1433; DATABASE=' +database+
                    ';UID=' +username+
                    ';PWD=' +password)

cursor = cnxn.cursor()

tsql = "SELECT * FROM iris2;"
with cursor.execute(tsql):
    row = cursor.fetchone()

    while row:
  
        print(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + " " +
                str(row[3]) + " " + str(row[4]))
        row = cursor.fetchone()
cnxn.close()
