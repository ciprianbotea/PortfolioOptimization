import pandas as pd
import mysql.connector as msql
from mysql.connector import Error

data = pd.read_csv('bvb.csv', delimiter=',')
bet = pd.read_csv('bet.csv', delimiter=',')

try:
    connection = msql.connect(host='localhost', database='bvb_stocks', user='root', password='')
    if connection.is_connected():
        cursor = connection.cursor()
        cursor.execute("select database();")
        record = cursor.fetchone()
        print("You're connected to database: ", record)
        cursor.execute('DROP TABLE IF EXISTS closing_prices;')
        print('Creating table....')

        cursor.execute(
            "CREATE TABLE closing_prices(date varchar(255), ALR float, ATB float, BIO float, BRD float, "
            "BVB float, CMP float, COTE float, DIGI float, EL float, EVER float, FP float, M float, "
            "ROCE float, SFG float, SIF1 float, SIF4 float, SIF5 float, SNG float, SNN float, "
            "SNP float, TEL float, TGN float, TLV float, TRANSI float, TRP float)")
        print("Table is created....")

        for i, row in data.iterrows():
            sql = "INSERT INTO bvb_stocks.closing_prices VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, " \
                  "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            cursor.execute(sql, tuple(row))
            print("Record inserted")
            connection.commit()

except Error as e:
    print("Error while connecting to MySQL", e)


try:
    connection = msql.connect(host='localhost', database='bvb_stocks', user='root', password='')
    if connection.is_connected():
        cursor = connection.cursor()
        cursor.execute("select database();")
        record = cursor.fetchone()
        print("You're connected to database: ", record)
        cursor.execute('DROP TABLE IF EXISTS bet_index;')
        print('Creating table....')

        cursor.execute(
            "CREATE TABLE bet_index(date varchar(255), BET float)")
        print("Table is created....")

        for i, row in bet.iterrows():
            sql = "INSERT INTO bvb_stocks.bet_index VALUES(%s, %s) "
            cursor.execute(sql, tuple(row))
            print("Record inserted")
            connection.commit()

except Error as e:
    print("Error while connecting to MySQL", e)
