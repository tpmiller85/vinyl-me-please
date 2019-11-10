import psycopg2
import numpy as np
import pandas as pd


class CustomerRetention(object):
    # This class will query a connected PostgreSQL database using psycopg2 and
    # process the data for further visualization and modeling.

    def __init__(self):
        self.conn = psycopg2.connect(database="vinyl", user="postgres",
                                     host="localhost", port="5435")
        self.cur = self.conn.cursor()

        # PostgreSQL query to be run on Vinyl Me, Please database. Enclose in
        # triple quotes for psycopg2: ''' '''
        self.db_query = '''SELECT * FROM subscription_statuses;'''

    # Run PostgreSQL query and return pandas DataFrame
    def get_retention_data(self):
        print("Executing query...\n")
        self.cur.execute(self.db_query)
        results = self.cur.fetchall()
        colnames = [desc[0] for desc in self.cur.description]
        self.df_ret = pd.DataFrame(results, columns=colnames)
        return self.df_ret

    def _activated_num(self, month):
        return sum(self.df_ret[self.df_ret['created_at']
                   < pd.to_datetime(month)]['status'] == 'Active')

    def _cancelled_num(self, month):
        return sum(self.df_ret[self.df_ret['created_at']
                   < pd.to_datetime(month)]['status'] == 'Cancelled')

    # Calculate customer retention rates on a monthly basis. Setting start_month
    # and/or end_month optional. Prints results and returns them as DataFrame.
    def monthly_cx_retention(self, start_month='2017-01', end_month='2019-09'):
        month_starts = pd.date_range(start_month, end_month, freq='MS')
        month_ends = pd.date_range(pd.to_datetime(start_month)
                         + pd.DateOffset(months=1), pd.to_datetime(end_month)
                         + pd.DateOffset(months=1), freq='MS')
        retention_list = []
        for start, end in zip(list(month_starts), list(month_ends)):
            ret = ((self._activated_num(end) - self._cancelled_num(end))
                   - (self._activated_num(end) - self._activated_num(start)))\
                   / (self._activated_num(start) - self._cancelled_num(start))\
                   * 100
            retention_list.append(ret)
            print(f"Customer Retention for {start:%Y-%m}: {ret:.1f}%")
        self.df_ret_rates = pd.DataFrame({'period_start_date': month_starts,
                                     'period_end_date': month_ends,
                                     'customer_retention_pct': retention_list})
        print("\nResults returned as pandas DataFrame")
        return self.df_ret_rates


if __name__ == '__main__':
    retention = CustomerRetention()
    retention.get_retention_data()
    retention.monthly_cx_retention()
