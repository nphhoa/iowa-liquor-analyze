# Iowa Liquor Sales Analysis and Customer Segmentation
## 🔍 Project Overview
### 📌 Objective
- This is a project based on wholesale liquor sales data for the state of Iowa in the United States with more than 2.6 million transactions in 2023. The Iowa government purchases liquor from suppliers and then distributes it to all retail or wholesale entities in the state. As a data analyst, I see this as a business operation issue for the Iowa state government to carry out the entire project.

- This project focuses on in-depth analysis of the Iowa wholesale liquor market, providing insights into purchasing trends, customer segments, and sales performance. This project aims to help the Iowa state government in particular make data-driven decisions and optimize its business strategy. At the same time, it can also be considered a sample analysis for individuals or businesses involved in the alcohol industry for their own business purposes.

- Raw liquor sales data was sourced from [Iowa Liquor Sales website](https://data.iowa.gov/Sales-Distribution/Iowa-Liquor-Sales/m3tr-qhgy/about_data).

### 📌 Business Goal
- Customer segmentation to provide more effective customer care strategies and gain a deeper understanding of their customers.
- Analyze sales trends by product, geography, and time to make recommendations for optimizing inventory, product allocation, and order quantities.
- Identify factors affecting sales and make recommendations to improve performance in second half of 2024.
- Provide market insights to support informed business decisions.

### 📌 Tools and Methodologies
- Analyze the data using Python for statistical computation and machine learning techniques.
  - Clean data: Using Python’s libraries like Pandas and Numpy.
  - Machine Learning: Using Python’s library Sklearn with KMeans algorithm.
  - Analysis: Using Python’s libraries like Scipy, Pandas and Numpy.
  - Visualization: Using Python’s libraries like Matplotlib, Seaborn.
- Use Power BI to build dashboard.
- Insights Presentation: Summarize my findings and insights based on the analysis of the data set.
- Analyze market trends and make recommendations.

## 💡 Domain Knowledge
### What is Customer Segmentation?
Customer segmentation (also known as market segmentation) is a marketing practice of dividing a customer base into sub-groups. It could be according to geographic, demographic, psychographic, behavioral, and other characteristics. The key to effective segmentation is to divide customers into groups based on; the prediction of their value to the business. After that target each group with different strategies in order to extract maximum value from both high and low-profit customers.

[Read more…](https://www.liveagent.com/academy/customer-segmentation/).

### What are RFM and K-Means in Customer Segmentation?
**RFM** stands for Recency, Frequency, and Monetary Value, and it is a technique used in marketing and customer segmentation to analyze and categorize customers based on their transaction behavior. Each of the three components has a specific meaning:
- Recency (R): How recently did the customer make a purchase?
- Frequency (F): How often does the customer make purchases within a specific timeframe?
- Monetary (M): How much money has the customer spent within a specific timeframe?
RFM Customer Segmentation helps businesses better understand their customers, target specific segments with tailored marketing efforts, enhance customer loyalty, and increase profitability through optimized marketing strategies.

**K-Means** is a clustering algorithm used for partitioning a dataset into a specified number of clusters based on the similarity of data points. When using K-Means with RFM analysis, you are essentially using the three RFM components as features to group similar customers into clusters. The algorithm aims to minimize the variance within each cluster and maximize the variance between clusters.
## 🛠 Approach and Process
### 1. Data Preparation and Cleaning
- About Dataset

| Column Name            | Description                                                                                                                        |
|------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| Invoice/Item Number    | Concatenated invoice and line number associated with the liquor order. This provides a unique identifier for the individual liquor products included in the store order |
| Date                   | Date of order                                                                                                                      |
| Store Number           | Unique number assigned to the store who ordered the liquor.                                                                        |
| Store Name             | Name of store who ordered the liquor.                                                                                              |
| Address                | Address of store who ordered the liquor.                                                                                           |
| City                   | City where the store who ordered the liquor is located                                                                             |
| Zip Code               | Zip code where the store who ordered the liquor is located                                                                         |
| Store Location         | Location of store who ordered the liquor. The Address, City, State and Zip Code are geocoded to provide geographic coordinates. Accuracy of geocoding is dependent on how well the address is interpreted and the completeness of the reference data used. Left NULL where unable to provide point location. |
| County Number          | Iowa county number for the county where store who ordered the liquor is located                                                    |
| County                 | County where the store who ordered the liquor is located                                                                           |
| Category               | Category code associated with the liquor ordered                                                                                   |
| Category Name          | Category of the liquor ordered.                                                                                                    |
| Vendor Number          | The vendor number of the company for the brand of liquor ordered                                                                   |
| Vendor Name            | The vendor name of the company for the brand of liquor ordered                                                                     |
| Item Number            | Item number for the individual liquor product ordered.                                                                             |
| Item Description       | Description of the individual liquor product ordered.                                                                              |
| Pack                   | The number of bottles in a case for the liquor ordered                                                                             |
| Bottle Volume (ml)     | Volume of each liquor bottle ordered in milliliters.                                                                               |
| State Bottle Cost      | The amount that Alcoholic Beverages Division paid for each bottle of liquor ordered                                                |
| State Bottle Retail    | The amount the store paid for each bottle of liquor ordered                                                                        |
| Bottles Sold           | The number of bottles of liquor ordered by the store                                                                               |
| Sale (Dollars)         | Total cost of liquor order (number of bottles multiplied by the state bottle retail)                                                |
| Volume Sold (Liters)   | Total volume of liquor ordered in liters. (i.e. (Bottle Volume (ml) x Bottles Sold)/1,000)                                          |
| Volume Sold (Gallons)  | Total volume of liquor ordered in gallons. (i.e. (Bottle Volume (ml) x Bottles Sold)/3785.411784)                                   |

- Overview
<img src="https://i.imgur.com/ardb5Lz.png">
- Load dataset into dataframe using Pandas
- Explore number of columns, rows, ranges of values
- Handle missing, incorrect and invalid data
- Perform any additional steps

### 2. Customer Segmentation Using Machine Learning

#### 2.1. Summary Dataset

<img src="https://i.imgur.com/271PULi.png">

#### 2.2. Calculate RFM
```
# Assign the current date (the date of performing the task) 
current_date = date(2024,7,1)

# Group by Customers and check last date of purchase
recency = df.groupby(by = 'Store Number')['Order Date'].max().reset_index()

# Change the data type of 'Order Date' and then rename the column.
recency['Order Date'] = pd.DatetimeIndex(recency['Order Date']).date
recency = recency.rename(columns = {'Order Date' : 'Last Purchase Date'})

# Calculate recency
recency['Recency'] = recency['Last Purchase Date'].apply(lambda x: (current_date - x).days)
recency.drop('Last Purchase Date', axis = 1, inplace = True)

# Group Customers by Invoice
frequency = df.groupby(by = 'Store Number')['Invoice Number'].count().reset_index()
frequency = frequency.rename(columns = {'Invoice Number' : 'Frequency'})
# Group Customers by Revenue
monetary = df.groupby('Store Number')['Sale (Dollars)'].sum().reset_index()
monetary = monetary.rename(columns = {'Sale (Dollars)' : 'Monetary'})
# Create RFM Table
rfm = recency.merge(frequency, on = 'Store Number')
rfm = rfm.merge(monetary, on = 'Store Number')
# Print dataframe
rfm
```
Result: 
| Store Number | Recency | Frequency | Monetary    |
|--------------|---------|-----------|-------------|
| 2106         | 4       | 4868      | 1265007.25  |
| 2130         | 4       | 4907      | 2385691.63  |
| 2190         | 1       | 16398     | 3184450.15  |
| 2191         | 5       | 10071     | 2217721.30  |
| 2200         | 6       | 4951      | 461698.06   |
| ...          | ...     | ...       | ...         |
| 10418        | 2       | 50        | 5615.68     |
| 10419        | 18      | 4         | 428.85      |
| 10420        | 5       | 129       | 14042.67    |
| 10422        | 3       | 30        | 5684.28     |
| 10429        | 4       | 275       | 30270.72    |

### 3. RFM Data Preparation

#### 3.1. Summary about RFM
<img src="https://i.imgur.com/fF1gMrD.png ">

<img src="https://i.imgur.com/l5bYakc.png">

<img src="https://i.imgur.com/3fRPnSd.png ">

Some problems:
- Recency distribution is right-skewed.
- Frequency distribution is right-skewed.
- Monetary distribution is right-skewed.
- There are relatively many outliers in all three variables above.
=> All three fields do not follow a normal distribution of data. It is necesssary to transformation data before using K-mean.

Some method for transformation:
- log transformation.
- square root transformation.
- box-cox transformation.
- cube root tranformation.

#### 3.2. Feature Engineering
```
def analyze_skewness(x):
    fig, ax = plt.subplots(1,5, figsize=(10,5))
    sb.distplot(rfm[x], ax=ax[0], color= "#005A74")
    sb.distplot(np.log(rfm[x]), ax=ax[1], color= "#005A74")
    sb.distplot(np.sqrt(rfm[x]), ax=ax[2], color= "#005A74")
    sb.distplot(stats.boxcox(rfm[x])[0], ax=ax[3], color= "#005A74")
    sb.distplot(np.cbrt(rfm[x]), ax=ax[4], color= "#005A74") 
    plt.tight_layout()
    plt.show()
    #Print result 
    print("Original Skewness:", rfm[x].skew().round(2))
    print("Log-transformed Skewness:", np.log(rfm[x]).skew().round(2))
    print("Square Root-transformed Skewness:", np.sqrt(rfm[x]).skew().round(2))
    print("Box-Cox transformed Skewness:", pd.Series(stats.boxcox(rfm[x])[0]).skew().round(2))
    print("Cube Root-transformed Skewness:", pd.Series(np.cbrt(rfm[x])).skew().round(2))

for col in rfm.columns[1:]:
    analyze_skewness(col)
```

#### 3.3. Create a new RFM df with boxcox
```
rfm_boxcox = pd.DataFrame()
# Tranform Recency to box-cox transformation
rfm_boxcox['Recency'] = stats.boxcox(rfm['Recency'])[0]

# Tranform Frequency to box-cox transformation
rfm_boxcox['Frequency'] = stats.boxcox(rfm['Frequency'])[0]

# Tranform Monetary to box-cox transformation
rfm_boxcox['Monetary'] = stats.boxcox(rfm['Monetary'])[0]

# Check new dataframe
rfm_boxcox
```

Result:

| Recency   | Frequency | Monetary  |
|-----------|-----------|-----------|
| 1.102422  | 14.782862 | 15.106021 |
| 1.102422  | 14.805060 | 15.840441 |
| 0.000000  | 18.419269 | 16.176332 |
| 1.235703  | 16.895403 | 15.755681 |
| 1.337254  | 14.829919 | 13.948927 |
| ...       | ...       | ...       |
| 0.616659  | 4.997237  | 9.024386  |
| 1.830947  | 1.508753  | 6.252162  |
| 1.235703  | 6.605561  | 10.029880 |
| 0.914770  | 4.203608  | 9.037644  |
| 1.102422  | 8.028662  | 10.879769 |

#### 3.4. Scaler data with standard scaler
```
# Using StandardScaler
scaler = SklearnTransformerWrapper(transformer = StandardScaler())
scaler.fit(rfm_bc)
rfm_scaler = scaler.transform(rfm_bc)
rfm_scaler
```

Result:

| Recency    | Frequency | Monetary  |
|------------|-----------|-----------|
| -0.552564  | 1.453189  | 1.844912  |
| -0.552564  | 1.461343  | 2.347522  |
| -2.731622  | 2.788803  | 2.577394  |
| -0.289118  | 2.229103  | 2.289516  |
| -0.088392  | 1.470473  | 1.053040

> 💡 Check summary RFM data when using box-cox and standard scaler
<img src="https://i.imgur.com/1C8RJGt.png">
<img src="https://i.imgur.com/unHAMnh.png">
<img src="https://i.imgur.com/WeXmrTZ.png">

After applying the Standard Scaler, the distributions of the three variables appear to be close to normal distributions. Additionally, the proportion of outliers in the variables has significantly decreased.

=> It is possible to proceed with using an algorithm to segment customers.

### 4. Customer segmentation
#### 4.1. Using K-Means with Elbow Method 
<img src="https://i.imgur.com/Ox5K3DH.png">

The elbow method might not be very helpful in this situation, but looking at the overall picture, we can make a preliminary choice with 3 to 5 clusters for the problem.

#### 4.2. Using Hierarchical Clustering with Dendrogram
<img src="https://i.imgur.com/dDjeW2s.png">

The dendrogram provides a clearer view of how the data is clustered and somewhat reinforces the choice between 3 or 4 clusters from the previous method.

#### 4.3. Using K-means with Silhouette Method
```
silhouette_scores = []
possible_k_values = range(2, 11)  # k from 2 to 10
for k in possible_k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_scaler)
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(rfm_scaler, labels))

# Visualize Silhouette Score for each k value
plt.figure(figsize=(10, 6))
plt.plot(possible_k_values, silhouette_scores, marker='o', color= "#005A74")
plt.title('Silhouette Score for Different k',loc = 'left',  fontweight = 'heavy', fontsize = 16)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.xticks(possible_k_values)
plt.grid(True)
plt.show()

# Choose the best k
best_k = possible_k_values[np.argmax(silhouette_scores)]
print(f"Best k value based on Silhouette Score: {best_k}")
```
Result:

<img src="https://i.imgur.com/GadyYH0.png ">

```
from yellowbrick.cluster import SilhouetteVisualizer
# Visualize
fig, ax = plt.subplots(2, 2, figsize=(15, 9))
for idx, k in enumerate([2, 3, 4, 5]):
  km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=100, random_state=42)
  q, mod = divmod(idx, 2)

  visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q][mod])
  visualizer.fit(rfm_scaler)
  ax[q][mod].text(0.5, -0.1, f'Silhouette Score: {np.mean(visualizer.silhouette_score_):.2f}', size=12, ha='center', transform=ax[q][mod].transAxes)
plt.tight_layout()
plt.show()
```

Result: 

<img src="https://i.imgur.com/le5fGv2.png">

In all three methods, we will not choose k = 2. Although the Silhouette Score indicates that k = 2 has the highest score, when looking at the Silhouette Chart at k = 2, the density of data within each cluster is relatively high. While the score and the consistency between clusters are reasonable, we need more than 2 customer groups in this problem to identify more detailed characteristics of each group, facilitating effective planning for the upcoming year.

For the Hierarchical Clustering method, we can more clearly determine the number of clusters, which are 3 and 4.

Finally, to make a decision based on the two points above after eliminating k = 2 and narrowing down the clusters to 3 and 4, we observe that the Silhouette Score at k = 3 is higher than the other clusters. However, after going through some detailed testing processes such as the number of customers in each cluster, the characteristics of the centroids of each cluster, k = 4 provides the optimal result among the mentioned clusters.

Therefore, by combining the three clustering methods as above, I choose k = 4.
#### 4.4. Choose k = 4 and fit it with scaler data (using K-Means)
<img src="https://i.imgur.com/QgXyKVi.png">

The four clusters are relatively well-separated. The data mixing ratio between clusters is quite low.

```
# Create Cluster featue for rfm data
rfm['Cluster'] = clusters
# Calculate mean score clusters for each feature
rfm_clusters = rfm.groupby('Cluster').agg( { 'Recency':'mean', 'Frequency':'mean', 'Monetary':'mean' } ).round(0)
rfm_clusters.reset_index(inplace = True)
rfm_clusters
```
Result:
| Cluster | Recency | Frequency | Monetary |
|---------|---------|-----------|----------|
| 0       | 2.0     | 1513.0    | 177209.0 |
| 1       | 11.0    | 1000.0    | 110327.0 |
| 2       | 100.0   | 257.0     | 33095.0  |
| 3       | 7.0     | 4780.0    | 976250.0 |

> 💡 Visualize with 3D graph by Recency, Frequency and Monetary for main data (rfm dataframe)
<img src="https://i.imgur.com/afwUpyC.png">

#### 4.5. Summary about RFM after clustering customer
```
fig = px.scatter(rfm_clusters, x = "Recency", y = "Monetary", size = "Frequency", color = "Cluster", hover_name = "Cluster", size_max = 100)  
fig.update_layout(height=500, width=1500)
fig.update_layout(
    title="Details for each Clusters with Mean RFM values",
    height=500,
    width=1500
)
fig.show()
```
Result: 
<img src="https://i.imgur.com/GztAO1I.png">

**Analysis of Customer Cluster Homogeneity and Differentiation:**

1. Customer Count:

Homogeneity:
- The number of customers in each cluster is relatively even, with no significant differences. This indicates that the clustering process is effective and the clusters are of similar size.

Benefits:
- Facilitates easier comparison and analysis of clusters.
- Ensures the representativeness of each cluster in the analysis.

2. Distinctive Characteristics:

Differentiation:
- The clusters are distinct from each other with their own unique characteristics. This indicates that the clustering process is effective and the clusters have clear differences in characteristics.

Benefits:
- Helps to clearly identify different customer groups.
- Supports the development of appropriate marketing strategies for each customer group.

3. No Data Overlap:

Separation:
- There is no overlap of data between the clusters. This indicates that the clustering process is accurate and customers are clearly classified into each cluster.

Benefits:
- Ensures the accuracy of the analysis.
- Avoids misclassification of customers into inappropriate clusters.

-> Conclusion:

The analysis shows that the customer clustering process wiht K-Means model is effective and the clusters have homogeneity in terms of customer count, differentiation in terms of characteristics, and no data overlap. This makes it easier and more effective to compare, analyze and develop marketing strategies for each customer group.

#### 4.6. View Customers Group Data Distribution

To make reading numbers and drawing conclusions through visual charts easier, we will use 90% of the data for observation.

<img src="https://i.imgur.com/qRlIeyh.png">

```
rfm_clusters
```
| Cluster | Recency | Frequency | Monetary |
|---------|---------|-----------|----------|
| 0       | 2.0     | 1513.0    | 177209.0 |
| 1       | 11.0    | 1000.0    | 110327.0 |
| 2       | 100.0   | 257.0     | 33095.0  |
| 3       | 7.0     | 4780.0    | 976250.0 |

**Project conclusion**

By using 90% of the data for observation, we now observe that the histogram plots better depict the distribution of the data. Combined with the Cluster Centers, we will label each customer segment and propose solutions for the following issues:

*Cluster 0:*
- Summary
  - Recency: The average time since the last purchase is 2 days. The time ranges from 1 to 3 days at a particular store. (Frequent)
  - Frequency: The average frequency of purchases is 1,513 orders per 1.5 years. The number of orders ranges from 500 to 2,500 (with a maximum of 1,000) orders at a particular store. (On average)
  - Monetary: The average expenditure is 177,209 (Dollars). (High)
- Potential Loyalist: As a potential loyal customer group, these are individuals who have made recent purchases, but their quantity and value of orders are only at average levels.
- Problem: How can we increase the value of their shopping cart with each purchase?
- Recommend:
  - Offer free gifts for transactions above the brand's average value.
  - Upsell high-value products.
  - Seek feedback and implement campaigns to enhance engagement.

*Cluster 1:*
- Summary:
  - Recency: The average time since the last purchase is 11 days. The time ranges from 4 to 14 days (with a maximum of 7 days) at a particular store. (Average)
  - Frequency: The average frequency of purchases is 1,000 orders per 1.5 years. The number of orders ranges from 250 to 1750 orders at a particular store. (Low)
  - Monetary: The average expenditure is 110,327 (Dollars). The expenditure ranges from 20,000 to 200,000 (Dollars). (Relatively high compared to the total number of orders)
- Big Spenders: This group of customers does not make purchases frequently, but they have a high value per order compared to the modest frequency of purchases.
- Problem: How can they be encouraged to make more frequent purchases?
- Recommend:
  - Offer limited-time special promotions.
  - Provide recommendations based on their previous purchases.
  - Encourage them to join a membership program to receive more benefits, such as reward points, special discounts, thereby increasing their purchase frequency and fostering brand loyalty.

*Cluster 2:*
- Summary:
  - Recency: The average time since the last purchase is 100 days. The time ranges from 50 to over 300 days at a particular store. (Quite long)
  - Frequency: The average frequency of purchases is 257 orders per 1.5 years. The number of orders ranges from 50 to 500 orders at a particular store. (Very Low)
  - Monetary: The average expenditure is 33,095 (Dollars). The expenditure ranges from 5,000 to 50,000 (Dollars). (Very Low)
- Need Attention: This group of customers purchases infrequently, with relatively low quantity and value of orders.
- Problem: What causes their dissatisfaction and infrequent purchases?
- Recommend:
  - Reconnect with these customers through various means such as emails, direct interactions via social media, or phone calls.
  - Implement retargeting campaigns, short-term promotional programs with forms of vouchers, discounts, exclusive offers, etc.
  - Offer free trial policies to encourage them to return and make purchases.
  - Additionally, analyze their shopping cart history to identify any product-related factors contributing to their dissatisfaction.

*Cluster 3:*
- Summary:
  - Recency: The average time since the last purchase is 7 days. The time ranges from 1 to 7 days at a particular store. (Regularly)
  - Frequency: The average frequency of purchases is 4780 orders per year. The number of orders ranges from 2,000 to 10,000 orders at a particular store. (Very high)
  - Monetary: The average expenditure is 976,250 (Dollars). The expenditure ranges from 500,000 to 2,000,000 (Dollars). (Very High)
- Champions: These are new customers who transact frequently and spend the most. They are highly loyal, generous spenders, and likely to make another purchase soon.
- Problem: How can we retain these customers by any means necessary?
- Recommend:
  - Offer them privileges such as special discounts or early access to new products.
  - Attract them with exclusive promotional discount programs.
  - Encourage them to participate in loyalty programs to receive more benefits from their shopping.

Following customer segmentation analysis, Iowa’s customers were divided into four segments using RFM to build K-means machine learning model.

Each segment will be characterized by three customer attributes:
- R - Number of days since the last purchase as of July 1, 2024.
- F - Total number of orders placed from 1/1/2023 to 30/6/2024.
- M - Total amount spent in 1/1/2023 - 30/6/2024.
<img src="https://i.imgur.com/hjgWetV.png">

**Customer Ratio Analysis by Group:**
1. Big Spenders Group:
Percentage: 44% of customers belong to the Big Spenders group.
Opportunity: This is the largest group, indicating a strong foundation of high-value customers who contribute significantly to revenue.
Strategy: Focus on personalized marketing and exclusive offers to maintain and increase their spending. Consider loyalty programs to encourage repeat purchases.
2. Champions Group:
Percentage: 23% of customers belong to the Champions group.
Status: This group is stable and crucial for long-term success.
Strategy: Maintain high levels of satisfaction through exceptional customer service and unique rewards. Ensure ongoing engagement to retain their loyalty.
3. Need Attention Group:
Percentage: 20% of customers belong to the Need Attention group.
Priority: This group needs to be addressed promptly to prevent churn.
Action: Conduct a detailed analysis to understand their issues and provide targeted solutions. Implement retention strategies to convert them into loyal customers.
4. Potential Loyalist Group:
Percentage: 13% of customers belong to the Potential Loyalist group.
Opportunity: This group has the potential to move up to the Big Spenders or Champions categories with the right engagement.
Strategy: Develop targeted campaigns to nurture loyalty and encourage higher spending. Provide incentives and personalized offers to enhance their experience.

🖇 See entire source code at RFM_analyze.

### Sales Analysis
#### 1. Overview of Iowa’s wholesale liquor situation
<img src="https://i.imgur.com/6WEeGEi.png ">

Observations:
- Revenue and Profit Relationship: With an average revenue of $37.0 million per month, the data indicates a significant overall revenue landscape for the retail industry.
- Revenue Peaks: The highest monthly revenue in the observed period ranged between $32 to $39 million.
- Correlation Between Revenue and Profit: There is a noticeable correlation between revenue and profit. Months with higher revenue, such as June and December, also show increased profit growth.
- Stable Profit Margins: Profit margins remained relatively stable throughout the period, consistently around 33%.

Trends:
- Overall Trend: Both monthly revenue and profit show a steady upward trend, indicating growing demand in the market.
- Seasonal Trend: There is an observable increase in revenue towards the end of the year, particularly in October and December. This could be attributed to seasonal factors such as holidays and increased consumer spending during these months.
- Profit Margin Stability: The profit margins are stable, suggesting operational efficiency and effective cost management within the industry.

Strategic Insights:
- Revenue Growth: The steady increase in revenue and profit highlights a positive market trend. Businesses should continue leveraging this growth with strategic marketing and sales initiatives.
- Seasonal Opportunities: The significant revenue peaks towards the end of the year suggest that businesses should prepare for increased demand during the holiday season by optimizing inventory and enhancing promotional activities.
- Maintain Efficiency: The stable profit margins reflect effective operational practices. Maintaining this efficiency will be crucial for sustaining profitability as revenue grows.
>💡While the overall trend in wholesale liquor sales in Iowa in 2023 is positive, there is room for further optimization of profit. A deeper dive into the monthly growth patterns reveals insights that can guide strategic decision-making.
#### 2. Overall Growth Trends in Wholesale Liquor Sales
*Overall Growth*
- General Trend: The chart shows an increasing trend in the growth rate throughout 2023, followed by fluctuations in 2024.
- Fluctuations: There are significant fluctuations in the monthly growth rate, indicating varying performance across different months in both 2023 and 2024.
*Peak Growth Periods*
- Highest Growth Rates in 2023: March (13.37%), May (21.57%), October (14.54%)
- Highest Growth Rates in 2024: April (10.79%), June (8.41%)

These periods align with seasonal demand spikes, such as spring break, Memorial Day, and the holiday season.

*Slowdown Periods*
- Sharpest Declines in Growth Rate in 2023: July (-12.97%), September (-11.8%)
- Sharpest Declines in Growth Rate in 2024: January (-8.21%), February (-7.34%), June (-14.12%)

*Seasonal Patterns:* The data highlights the influence of seasonality on wholesale liquor sales. Growth rates were higher in the middle and end of the year (March, May, October) compared to the summer months (July, September).
<img src="https://i.imgur.com/2ADeJZO.png">

The provided chart and analysis clearly demonstrate the significant impact of seasonality on wholesale liquor sales in Iowa. The data reveals distinct patterns and trends associated with different seasons.
*Key Seasonal Trends:*
- Summer Surge: Sales consistently peak during the summer months (June-August), with revenue exceeding $200 million in 2023 and half of 2024.
- Winter Boost: Sales experience a secondary peak during the winter holiday season (November-December). Although the data is incomplete for winter 2024, current figures show a revenue of about $120 million, indicating significant consumption.
- Spring and Fall Lulls: Sales generally dip during spring (March-May) and fall (September-October).

*Factors driving seasonal trends:*
- Summer:
  - Warm weather, many people participate in outdoor activities, travel, picnics, leading to increased demand for alcohol consumption in these activities.
  - Holidays and festivals such as Memorial Day, Independence Day, Labor Day encourage gatherings, eating, and drinking.   
- Winter:
  - Cold weather makes people tend to stay at home, gather with friends and family, bringing the possibility of using alcohol to keep warm.
  - Major holidays close together like Thanksgiving, Christmas, and New Year's Day are also occasions for people to consume and push alcohol sales to the highest level in winter compared to the other three seasons of the year. In addition, on these major holidays, in addition to personal consumption needs, there is also the use of alcohol as gifts, which is when higher quality (higher priced) wines are preferred. 
- Spring and Fall: The weather is somewhat mild, with fewer holidays, leading to decreased demand for alcohol.

-> It can be seen that after a series of long major holidays in winter that drive up alcohol consumption, spring is when people focus on getting back to work for the new year, and from there the demand for alcohol decreases significantly. Then the market gradually stabilizes in the two mid-seasons (not much difference between summer and autumn) and starts to increase strongly again in winter. This is like the alcohol consumption cycle in Iowa that the dataset shows.

*Recommendations:*
- Inventory Management: Anticipate seasonal demand fluctuations by adjusting inventory levels. Increase stock during peak seasons (summer and winter) and reduce stock during slower periods (spring and fall).
- Targeted Marketing: Implement targeted marketing campaigns and promotions during off-peak seasons (spring and fall) to stimulate sales and attract customers.
- Product Assortment: Offer a diverse selection of seasonal and holiday-themed alcoholic beverages to cater to specific consumer preferences during different times of the year.
- Pricing Strategies: Consider implementing dynamic pricing strategies that adjust prices based on seasonal demand and customer behavior.
- Data-Driven Analysis: Continuously monitor sales data and consumer trends to refine seasonal forecasting and optimize inventory management practices.

By understanding and effectively managing seasonal trends, wholesale liquor distributors in Iowa can maximize sales, optimize inventory costs, and maintain a competitive edge throughout the year. Leveraging data-driven insights and implementing strategic initiatives can lead to improved profitability and long-term success.

#### 3. Fluctuations between order quantity and average order value
<img src="https://i.imgur.com/k1AlUEt.png">
There were a total of 3885075 wine order, of which:
- Overall Trend: The total number of invoices shows a gradual increase from the beginning of the year, with notable spikes in mid-year (particularly in June) and again towards the end of the year (December).
- Correlation: There is a clear correlation between revenue and order volume. Months with higher invoice numbers tend to coincide with increased revenue, suggesting a linear relationship between these two metrics.

<img src="https://i.imgur.com/nVQfl5k.png">

The data reveals a contrasting trend between revenue and order value in wholesale liquor sales in Iowa. 
There are noticeable peaks, particularly in early 2023 (February) and again in mid-2023 (June), with the highest average revenue reaching around $180.

Unlike the order volume, the average order value does not exhibit a strong seasonal dependency, indicating that while order frequency may increase during peak seasons, the average value per order remains consistent.

*Key Takeaways:*
- Revenue and Order Volume: Revenue growth is driven by increased order volume, not by higher average order value.
- Seasonal Patterns: Order volume increases during peak seasons (summer and winter), while average order value remains consistent.
- Customer Behavior: Customers tend to place more frequent orders with smaller average values, likely due to inventory management and cost considerations.

*Potential Reasons for Stable Average Order Value:*
- Inventory Management: Customers, particularly retailers, may prefer to place smaller orders more frequently to minimize inventory costs and storage requirements.
- Cost Considerations: Smaller orders may reduce transportation and storage costs, especially for customers with limited storage space or budgets.
- Purchasing Habits: Customers may have established purchasing patterns that prioritize smaller orders over larger ones.

*Implications and Challenges:*
- Increased Order Processing Costs: Higher order volume, despite stable average order value, can lead to increased order processing costs, such as labor and system overhead.
- Transportation and Storage Costs: While smaller orders may reduce costs for customers, they can increase transportation and storage costs per unit for distributors.

*Recommendations:*
- Order Value Analysis: Analyze order value distribution throughout the year to identify potential thresholds for differentiating shipping costs.
- Seasonal Incentives: Implement seasonal incentives, such as discounts or free shipping, to encourage customers to increase order value during peak seasons.
- Minimum Order Requirements: Consider implementing minimum order requirements or tiered pricing structures to encourage larger orders and offset increased processing costs.
- Customer Segmentation: Segment customers based on order behavior and preferences to tailor marketing and incentive strategies accordingly.

Understanding the relationship between revenue, order volume, and average order value is crucial for wholesale liquor distributors in Iowa. By analyzing these trends and implementing targeted strategies, distributors can optimize inventory management, reduce costs, and drive profitable growth.

#### 4. Customer behavior by month
<img src="https://i.imgur.com/7z8MSSX.png">
We currently have a total of 2,158 customers. The number of active customers per month remains relatively high throughout 2023 and the before half of 2024, ranging from approximately 1,800 to 1,900 customers per month.

The positive aspect above is largely due to the fact that the majority of our customers are retail stores in the state. Additionally, as we are the only wine distributor in the state, most of the stores that sell wine will continue to consume our wine for most of their business lifecycle.

<img src="https://i.imgur.com/NRI9Xvv.png">

Due to the nature of the customer file in the wholesale sector, the rate of customers returning is always very high (93% - 96%).

Customer retention in Iowa shows relatively high stability, with some minor fluctuations. Strong growth in June 2023 and June 2024 is a positive sign, however, continued improvement in strategies during low retention months is needed to maintain stability and sustainable growth.

This could be predicted for the following reasons:
- February to May 2023: The decline may be due to factors not included in the data such as seasonality, reduced consumer spending, or ineffective marketing campaigns.
- June 2023: The sharp increase may be the result of more effective marketing campaigns or special events driving consumption.
- July to December 2023: The stability in customer retention rates suggests that Iowa's wine distribution system is operating effectively.
- January to June 2024: Steady growth, especially in June 2024, may be due to improved customer service strategies or effective promotions.

#### 5. Top 5 Customers with Highest and Lowest Profit
<img src="https://i.imgur.com/YNQNR8E.png">
HY-VEE #3 and CENTRAL CITY 2 are the top two profit-generating customers, exceeding the 8 million dollar from 2023 to 30/6/2024.

<img src="https://i.imgur.com/RGj6dKv.png">

On the other hand, there are customers who seem to only purchase once and show no signs of returning, as evidenced by the extremely low profit generated from these customers. Examples include SOUTHERN GLAZERS WINE & SPIRITS with less than 50 and Makil's #2 with aroud 150.

#### 6. Revenue and Profit Trends by Customer Segment
<img src="https://i.imgur.com/hZ2GqwH.png">

An analysis of the two charts reveals a strong adherence to the Pareto principle, where 23% of customers, categorized as "Champions," generate 74% of the total profit. While the overall business performance is positive with a consistent upward trend, stable monthly active customers, and a high retention rate, the reliance on a small group of high-profit customers highlights the need to improve profitability across the customer base.

*Key Takeaways:* 
- Pareto Principle Application: The data aligns with the Pareto principle, indicating that a small percentage of customers (Champions) contribute significantly to overall profitability.
- Profit Concentration: The concentration of profit among a limited group of customers suggests opportunities to expand profitability across the broader customer base.
- Customer Segmentation Analysis: Detailed analysis in the Customer Segmentation Analysis section provides insights into specific issues and potential solutions for each customer segment.

#### 7. Top 5 Best-selling products

<img src="https://i.imgur.com/9QyQGML.png">

A total of 5,701 different products were sold in 2023 and the first half of 2024, with a combined total of over 46 million bottles of wine. However, 25% of these products (equivalent to 1,313 products) sold less than 48 bottles in the year.

On the other hand, FIREBALL CINNAMON WHISKEY 100.0 (ml) is considered the product of the year in 2023 and 1/2 of 2024, selling over 3.5 million bottles (3.5 times more than the next closest product), far outperforming the rest of the products.

Some suggestions in this analysis are as follows:
- Consider discontinuing the import of liquors belonging to the bottom 25% group (those selling less than 36 bottles in a year) to save warehousing costs.
- Conduct an in-depth analysis of the top-selling liquors (ingredients, flavor, price, etc.) to find similar products to import in order to maximize profit per product.
#### 8. Product Consumption Trends
<img src="https://i.imgur.com/uZS5cRc.png">

The analysis of the relationship between product diversity, profit, and product sales reveals a relatively weak correlation between these variables. This suggests that product diversification does not significantly drive profit growth. Instead, customer preference appears to be concentrated on a limited subset of products, roughly one-fifth of the total product portfolio.

*Key Takeaways:*
- Weak Correlation: The scatterplot indicates a weak positive correlation between product diversity and profit, suggesting that increasing the number of products does not directly translate into substantial profit gains.
- Customer Preference: The concentration of data points implies that customers primarily focus on a smaller subset of products, indicating that product diversity beyond this range may not be fully utilized by customers.
- Optimization Strategies: This analysis suggests that focusing on optimizing the performance of the most popular and profitable products might be more effective than expanding the product portfolio indefinitely.

<img src="https://i.imgur.com/6PmHnQI.png">

The analysis suggests that focusing on popular product categories, such as VODKA, MINI, BOURBON, REPOSADO, WHISKEY, PET, can be an effective strategy to enhance import efforts and attract new customers. By understanding the characteristics that appeal to consumers of these products, wholesale liquor distributors in Iowa can expand their product offerings, optimize customer acquisition, and maximize profitability.

*Key Takeaways:*
- Popular Product Categories: Prioritize importing and promoting products within the Bourbon, Vodka, Whiskey, and Reposado categories, as these align with strong consumer demand.
- Consumer Attraction: Analyze the characteristics that attract consumers to these popular product categories to identify common preferences and trends.
- Product Diversification: Utilize insights from consumer preferences to expand the product portfolio with similar offerings, catering to a wider range of customer tastes and interests.
- Profit Optimization: Focus on optimizing the profitability of popular product categories by implementing effective pricing strategies, targeted marketing campaigns, and efficient inventory management.
#### 9. Liquor market in the state of Iowa

<img src="https://i.imgur.com/1bRK5HQ.png">

Our customer file spans across 99 counties in the state of Iowa. However, the number of stores is influenced by geographical location with 50% concentrated in just 13 counties.

Among these, Polk County has the highest concentration of stores with 287, followed by Linn County with 150, far exceeding the number of stores in the rest of the state.

Polk County is located in the U.S. state of Iowa. According to the 2020 census, the population was 492,401. It is the most populous county in Iowa and home to over 15% of the state's residents. The county seat is Des Moines, which is also the capital city of Iowa. This partly explains why it has the highest concentration of stores in the state.

<img src="https://i.imgur.com/IMunChv.png">

Profit is indeed influenced by geographical location. In central districts with high population density and a good standard of living, a large portion of stores are distributed to meet the high demand for living needs. The concentration of stores in these "Gold" areas drives sales and increases profits in a positive linear relationship.

However, retail/wholesale prices are not geographically dependent. The correlation coefficients are only at an average level, indicating that we maintain relatively stable selling prices that are not overly influenced by external factors. This is also a good sign for customer retention.
#### 10. Conclusion

**Positive Trends**
- Seasonal Revenue Growth: Revenue experiences seasonal growth in summer and winter. Profit margins remain consistently high at 33% throughout the year.
- Steady Revenue Growth Rate: The average monthly revenue growth rate for 2023 and the first half of 2024 was 1.35%. Although not extremely high, this is a positive sign considering the global economic downturn.
- High Order Volumes: There were a total of 3,885,075 liquor orders, with volumes peaking in summer and winter.
- Stable Average Order Value: The average order value remains stable throughout the year, with only a small variation of around $30 between the highest and lowest months.
- Active Customer Base: There are 2,158 customers across the state, with a consistently high number of active customers each month.
- Profitable Champions Group: 23% of customers (the Champions group) generated 74% of profits, indicating a highly effective customer segment.
- Diverse Product Range: The store offers 5,701 different products and has sold over 46 million bottles of liquor. Certain types of liquor, such as Vodka, Mini, Bourbon, Reposado, Whiskey, and PET, bring in high profits.
- Statewide Customer Reach: Customers are spread across all 99 counties in Iowa, suggesting high product recognition and potential for future expansion outside the state.

**Problems**
- Periods of Negative Growth: Negative growth was observed in four periods over two years: July (-12.97%) and September (-11.8%) of 2023, and January (-8.21%) and June (-14.12%) of 2024. Post-holiday and shopping season slumps contribute to this decline.
- High Costs Due to Seasonal Fluctuations: High order volumes in summer and winter increase transportation and warehousing costs due to numerous small orders, leading to linear cost growth alongside revenue.
- Underutilized Customer Groups: Besides the Champions group, the other three customer groups generate only 26% of profits, indicating untapped potential among the 2,158 existing customers.
- Inefficient Product Storage: 25% of the 5,100 types of liquor sold fewer than 39 bottles each, while the most consumed type in one store was 2,801 bottles. This results in high storage and inventory costs.
- Geographical Profit Discrepancies: Although liquor is distributed across all counties, 50% of customers are concentrated in just 13 counties, with Polk County having the most at 287 stores. This increases shipping costs to distant counties without affecting prices.

**Recommendations**
- Seasonal Adaptation: Increase orders in summer and winter to meet high demand and reduce orders during low consumption periods to lower inventory and storage costs.
- Optimize Order Value:
  - Analyze annual order values to determine a cost-effective threshold. Orders below this threshold incur higher shipping costs.
  - Introduce special policies during growth periods (summer and winter) to boost order values, such as discounts or free shipping.
- Streamline Product Portfolio:
  - Reduce imports of liquors in the bottom 25% (those selling fewer than 39 bottles) to save on storage costs.
  - Conduct in-depth analysis of top-selling liquors to identify similar high-profit imports.
- Expand Customer Base: Since most customers are in central counties or the capital, expanding the customer base outside Iowa could significantly increase revenue and customer numbers.

By addressing these challenges and implementing the recommended strategies, the wholesale liquor distributor in Iowa can enhance operational efficiency, optimize inventory management, improve customer engagement, and achieve sustainable growth.

## Dashboard

- Sales Analysis: Control the overall business effectiveness through keymetrics.
<img src="https://i.imgur.com/Zlma6yT.png">

- Customer Analysis: Assess the business effectiveness of each customer and different customer groups.
<img src="https://i.imgur.com/A6ZhdPH.png">

- Product Analysis: Detailed description of the performance of each product.
<img src="https://i.imgur.com/9zgBIBy.png">

