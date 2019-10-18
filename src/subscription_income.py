import psycopg2
import numpy as np
import pandas as pd


class SubscriptionIncome(object):
    # This class will query a connected PostgreSQL database using psycopg2, and
    # then clean the data for modeling.

    def __init__(self):
        self.conn = psycopg2.connect(database="vinyl", user="postgres",
                                host="localhost", port="5435")
        self.cur = self.conn.cursor()
        
        # PostgreSQL query to be run on Vinyl Me, Please database. Enclose in
        # triple quotes for psycopg2: ''' '''
        self.db_query = '''
            -- Query to return all non-zero subscription payment dates for all
            -- customers:
            SELECT
                shoppe_orders.received_at,
                shoppe_order_items.price
                FROM (
                    SELECT
                        id,
                        status,
                        received_at
                        FROM shoppe_orders
                        WHERE status = 'received'
                        ) AS shoppe_orders
                LEFT JOIN (
                    SELECT
                        order_id,
                        ordered_item_id,
                        quantity,
                        quantity * unit_price AS price
                        FROM shoppe_order_items
                        ) AS shoppe_order_items
                    ON shoppe_order_items.order_id = shoppe_orders.id
                LEFT JOIN (
                    SELECT
                        id,
                        sku,
                        price
                        FROM shoppe_products
                        ) AS shoppe_products
                    ON shoppe_products.id = shoppe_order_items.ordered_item_id
                WHERE shoppe_products.sku LIKE '%renewal%'
                AND shoppe_order_items.quantity > 0
                AND shoppe_order_items.price > 0
                ORDER BY shoppe_orders.received_at DESC;
                '''

    def get_subscription_income(self):
        # Execute SQL query and return pandas dataframe.
        print("Executing query...\n")
        self.cur.execute(self.db_query)
        results = self.cur.fetchall()
        colnames = [desc[0] for desc in self.cur.description]
        self.df_subscr_income = pd.DataFrame(results,
                                             columns=colnames)
        self.df_subscr_income['price'].astype('int32')
        self.df_subscr_income = self.df_subscr_income.groupby(
                               [self.df_subscr_income['received_at']
                               .dt.to_period("M")]).sum()
        print(self.df_subscr_income)
        print("\nResults returned as pandas DataFrame")
        return self.df_subscr_income


if __name__ == '__main__':
    subscr_income = SubscriptionIncome()
    subscr_income.get_subscription_income()
