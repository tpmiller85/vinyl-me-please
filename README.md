# Vinyl Me, Please

[![](images/vinyl_logo.jpg)](https://www.vinylmeplease.com)


## Table of Contents  
* [Overview & Goals](##overview)<BR>
* [Data Pipeline](##data_pipeline)<BR>
* [Exploratory Data Analysis (EDA)](##eda)<BR>

* [Conclusion and Future Project Ideas](##conclusion_future)<BR>


<a name="#overview"></a>

## Overview & Goals  

In their own words, Denver-based Vinyl Me, Please. is "a record of the month club. The best damn record club out there, in fact." They work with artists, labels and production facilities to re-issue old records, as well as release new albums. Their business model includes both monthly record club subscriptions as well as individual record sales. They have a large number of unique releases, and there are three 'release tracks' that users can subscribe to: Essentials, Classiscs, and Rap & Hip-Hop.

[![](images/gorillaz_records.jpg)](https://www.vinylmeplease.com)
[![](images/subscription_tracks.png)](https://www.vinylmeplease.com)

I had the privilege of working with their production database. Going into this project, my goals included the following:
* Work with the team at ‘Vinyl Me, Please’ to provide meaningful insights for their business.
* Gain experience wrangling messy, real-world data, including building a data pipeline from a SQL database to python data analysis and visualization tools that can be adapted to future
uses.
* Gain experience with modeling techniques to provide relevant business insights:
   * Are there album attributes that might contribute to the popularity of a certain release within the ‘Vinyl Me, Please’ ecosystem?
   * What insights into customer behavior can be gained from this data set?


<a name="#data_pipeline"></a>

## Data Pipeline  

The team at Vinyl Me, Please. gave me access to a 13GB PostgreSQL database dump. This production/sales database consists of 118 separate tables, which relate to various customer, product and transaction history details. In order to work with the database, I restored the DB dump into a PostgreSQL database running in a Docker container on my local system. From there, I established a pipeline to Python using psycopg 2.

Working in both PostgreSQL and Python allowed me to choose which language would most easily be able to handle a given task. On the SQL side, I built queries that joined up to five tables, and made use of convenient SQL aggregation functions.

<a name="#eda"></a>

## Exploratory Data Analysis (EDA)]

A significant amount of effort went into exploring all of the tables within the database, and mapping out which tables would need to be joined together in order to provide interesting insights. For example, the 'releases' table includes 36 columns of information for each album release, but the full album name is listed in the 'shoppe_product_translations' table, and sales information is in the 'product_sales_rollups' table. In the end, a 3-way table join provided the following results for each release in the database:

```SQL
-[ RECORD 1 ]-+-----
max_sales     | 24559
product_id    | 829
name          | Gorillaz 'Demon Days'
exclusive     | 0
download_code | f
color         | Red Translucent
lp_count      | 2
weight        | standard
release_year  | 2005
numbered      | f
jacket_type   | Gatefold
jacket_style  | Direct-To-Board
```

I built Python classes to perform my PostgreSQL queries via psycopg2 and then clean and/or process the resuls.


### Helpful PostgreSQL Utilities

I will mention two PostgreSQL utility queries that proved to be invaluable when dealing with such a large database. The following two queries can be seen in src/sql_utility_queries.txt:
* List all tables sorted by size
* Find all tables that contain a certain column


![](images/customer_count_dollars_spent.png)

![](images/subscription_income.png)

![](images/retention.png)

![](images/max_90_sales.png)
![](images/heatmap.png)



* Customer Count per Total Spent
* Subscription Renewal Income By Month
* Customer Retention Chart
* Album attributes corellated with MAX(sales.quantity_90d)

['Is this release exclusive to Vinyl Me, Please?',
                        'Is there a digital download code included with this record?',
                        'How many vinyl records are included in this release?',
                        'Is this a heavier weight pressing (180g or 200g)?',
                        'Is this a numbered/limited-run release?',
                        'Is this a Tip-On jacket, where the cover print is done on a separate sheet of text-stock paper, and then wrapped/glued to a thick corrugated core?',
                        'gatefold',
                        'color',
                        'recent']



* Plot Total # Sales vs. Releases
* Cancelations per month?

"Describe the process you used to ensure your model was properly fit."

discussion of the cleaning and featurization pipeline and how raw data were transformed to the data trained on.