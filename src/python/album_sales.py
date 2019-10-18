import psycopg2
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression


class AlbumSales(object):
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
                sales.max_sales,
                sales.product_id,
                s_prods.name,
                releases.exclusive,
                releases.download_code,
                releases.color,
                releases.lp_count,
                releases.weight,
                releases.release_year,
                releases.numbered,
                releases.jacket_type,
                releases.jacket_style
                    FROM (
                    SELECT MAX(sales.quantity_90d) AS max_sales,
                        sales.product_id
                    FROM product_sales_rollups AS sales
                    GROUP BY sales.product_id
                ) sales
                LEFT JOIN shoppe_product_translations AS s_prods
                    ON sales.product_id = s_prods.shoppe_product_id
                LEFT JOIN releases
                    ON releases.product_id = sales.product_id
                ORDER BY sales.max_sales DESC;
                '''

    def get_sales_data(self):
        # Execute SQL query above and return a pandas dataframe
        print("Executing query...\n")
        self.cur.execute(self.db_query)
        results_sales = self.cur.fetchall()
        colnames_sales = [desc[0] for desc in self.cur.description]
        self.df_sales = pd.DataFrame(results_sales, columns=colnames_sales)
        return self.df_sales

    def clean_sales_data(self):
        # Remove Membership Renewal items
        df_record_sales = self.df_sales[~self.df_sales['name']
                              .str.contains("embership")]

        # Cleaning steps to numerically encode all columns of interest

        # Creating tip_on and gatefold binary columns
        df_record_sales['jacket_text'] = df_record_sales['jacket_style'] \
                                         + df_record_sales['jacket_type']
        df_record_sales['jacket_text'].fillna('none', inplace=True)
        df_record_sales['tip_on'] = np.where(df_record_sales['jacket_text']
                                    .str.lower().str.contains('tip'), 1, 0)
        df_record_sales['gatefold'] = np.where(df_record_sales['jacket_text']
                                      .str.lower().str.contains('gate'), 1, 0)

        # Encode custom vinyl color as custom_color
        df_record_sales['color'].fillna('black', inplace=True)
        df_record_sales['custom_color'] = np.where(df_record_sales['color'].str
                                           .lower() != 'black', 1, 0)

        # Binary encode recent releases based on cutoff
        recent_cutoff_year = '2016'
        df_record_sales['release_year'] = pd.to_datetime(df_record_sales
                                           ['release_year'], errors='coerce')
        df_record_sales['recent_release'] = np.where(df_record_sales
                                          ['release_year'] > pd.to_datetime(
                                          recent_cutoff_year + '-01-01'), 1, 0)

        # Binary encode vinyl weight as 0 for standard and 1 for > 180g
        df_record_sales['weight'].fillna(0, inplace=True)
        df_record_sales['weight'] = df_record_sales['weight'].apply(
            lambda x: 0 if str(x).lower() == 'standard' else x)
        df_record_sales['weight'] = df_record_sales['weight'].apply(
            lambda x: 1 if str(x) == '180' else x)
        df_record_sales['weight'] = df_record_sales['weight'].apply(
            lambda x: 1 if str(x) == '200' else x)
        df_record_sales['weight'] = df_record_sales['weight'].apply(
            lambda x: x if x == 1 else 0)

        # Binary encode misc. columns
        df_record_sales['download_code'] = df_record_sales['download_code']\
                                         .apply(lambda x: 1 if x == True else 0)
        df_record_sales['exclusive'] = df_record_sales['exclusive']\
                                         .apply(lambda x: 1 if x == True else 0)
        df_record_sales['lp_count'].fillna(1, inplace=True)
        df_record_sales['numbered'] = np.where(df_record_sales['numbered']
                                                == True, 1, 0)

        # Dropping columns that are no longer needed
        self.df_records_drop = df_record_sales.drop(columns=['color',
                               'jacket_type', 'jacket_style', 'jacket_text',
                               'release_year'])

        print("\nResults returned as pandas DataFrame")
        return self.df_records_drop

    def regression_model(self):
        # Build a linear regression model based on the cleaned dataframe. Drop
        # columns that were previously found to be insignificant.
        self.df_model = self.df_records_drop.drop(columns=['product_id',
                                                           'name'])
        self.X = self.df_model.drop(columns=['max_sales'])
        self.y = self.df_model['max_sales']
        self.X_drop = self.X.drop(columns=['exclusive', 'download_code',
                                           'weight'])
        linear_model_drop = LinearRegression()
        linear_model_drop.fit(self.X_drop.to_numpy(),
                              np.log(self.y.to_numpy()+1))
        self.coef_df_drop = pd.DataFrame({'Feature': self.X_drop.columns,
                                          'Log Sales Impact (# sold)':
                                          linear_model_drop.coef_},
                                          index=None).round(2)
        self.coef_df_drop = self.coef_df_drop.sort_values(
                              by=['Log Sales Impact (# sold)'], ascending=False)

        return self.coef_df_drop


if __name__ == '__main__':
    sales = AlbumSales()
    sales.get_sales_data()
    sales.clean_sales_data()
    # not automatically running regression_model() here for convenience
