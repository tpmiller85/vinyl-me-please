import psycopg2
import numpy as np
import pandas as pd


class SpendingPerCustomer(object):
    # This class will query a connected PostgreSQL database using psycopg2, and
    # then clean the data for modeling.

    def __init__(self):
        self.conn = psycopg2.connect(database="vinyl", user="postgres",
                                     host="localhost", port="5435")
        self.cur = self.conn.cursor()

        # PostgreSQL query to be run on Vinyl Me, Please database. Enclose in
        # triple quotes for psycopg2: ''' '''
        self.db_query = '''
            SELECT
                customer_id,
                SUM(amount_paid) AS cx_total_spent
                FROM shoppe_orders
                GROUP BY customer_id
                ORDER BY cx_total_spent DESC;
                '''

    def get_spending_per_customer(self):
        # Execute SQL query and return pandas dataframe.
        print("Executing query...\n")
        self.cur.execute(self.db_query)
        results = self.cur.fetchall()
        colnames = [desc[0] for desc in self.cur.description]
        self.df = pd.DataFrame(results, columns=colnames)
        self.df = self.df.dropna()
        self.df['cx_total_spent'] = self.df['cx_total_spent'].astype('int32')
        print(self.df.head(5))
        print("\nResults returned as pandas DataFrame")
        return self.df


if __name__ == '__main__':
    spending = SpendingPerCustomer()
    spending.get_spending_per_customer()
