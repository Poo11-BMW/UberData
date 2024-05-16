#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[3]:


# Load the Uber dataset
df = pd.read_csv('uber_trip_data.csv')


# # DATA PREPROCESSING 

# In[4]:


# Display basic information about the dataset
print(df.info())


# In[5]:


# Convert date_time to datetime format
df['date_time'] = pd.to_datetime(df['date_time'])


# In[6]:


# Summary statistics
print(df.describe())


# In[7]:


# Data Cleaning and Preprocessing
#Removing missing values
df = df.dropna()


# In[8]:


# Ensure numerical fields are correctly typed and handle missing values
df['trip_duration'] = pd.to_numeric(df['trip_duration'], errors='coerce')
df['trip_distance'] = pd.to_numeric(df['trip_distance'], errors='coerce')
df['fare_amount'] = pd.to_numeric(df['fare_amount'], errors='coerce')


# # Removing Outliers 

# In[9]:


# Define columns containing numeric data
numeric_columns = ['trip_duration', 'trip_distance', 'fare_amount']

# Calculate z-score for each numeric column
z_scores = stats.zscore(df[numeric_columns])

# Define threshold for z-score to identify outliers (e.g., z-score > 3 or < -3)
threshold = 3

# Find rows where any of the z-scores exceed the threshold
outlier_rows = np.any(np.abs(z_scores) > threshold, axis=1)

# Remove outlier rows from the dataset
df_no_outliers = df[~outlier_rows]

# Display the shape of the original and filtered dataset to see the effect of outlier removal
print("Original dataset shape:", df.shape)
print("Dataset shape after removing outliers:", df_no_outliers.shape)


# In[10]:


# Define columns containing numeric data
numeric_columns = ['trip_duration', 'trip_distance', 'fare_amount']

# Calculate IQR for each numeric column
Q1 = df[numeric_columns].quantile(0.25)
Q3 = df[numeric_columns].quantile(0.75)
IQR = Q3 - Q1

# Define lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out rows containing outliers
outlier_mask = ((df[numeric_columns] < lower_bound) | (df[numeric_columns] > upper_bound)).any(axis=1)
df_no_outliers = df[~outlier_mask]

# Plot the distribution of numeric columns before and after removing outliers
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
sns.boxplot(data=df[numeric_columns])
plt.title('Before Removing Outliers')
plt.subplot(1, 2, 2)
sns.boxplot(data=df_no_outliers[numeric_columns])
plt.title('After Removing Outliers')
plt.tight_layout()
plt.show()


# In[11]:


print("Data Info After Conversion:")
print(df.info())


# # Analysis

# In[12]:


total_trips = len(df)
print("Total number of trips:", total_trips)


# In[13]:


# Calculate the total number of trips
unique_trip = df['trip_id'].nunique()
# Print the total number of trips
print("Number of unique trips:", unique_trip)


# In[12]:


# Find the number of unique drivers
unique_drivers = df['driver_id'].nunique()
# Print the number of unique drivers
print("Number of unique drivers:", unique_drivers)


# In[13]:


# Extract the day of the week
df['day_of_week'] = df['date_time'].dt.day_name()

# Plot histogram of trips per day of the week
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='day_of_week', data=df, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title('Number of Trips per Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Trips')
plt.xticks(rotation=45)

# Add counts above the bars
for p in ax.patches:
    ax.annotate(format(int(p.get_height()), ',d'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 9), 
                textcoords='offset points')

plt.show()


# In[14]:


# Extract the month
df['month'] = df['date_time'].dt.month_name()

# Plot histogram of trips per month
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='month', data=df, order=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
plt.title('Number of Trips per Month')
plt.xlabel('Month')
plt.ylabel('Number of Trips')
plt.xticks(rotation=45)

# Add counts above the bars
for p in ax.patches:
    ax.annotate(format(int(p.get_height()), ',d'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 9), 
                textcoords='offset points')

plt.show()


# In[15]:


# Extract the hour
df['hour'] = df['date_time'].dt.hour

# Plot histogram of trips per hour
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='hour', data=df, palette='viridis')
plt.title('Number of Trips per Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Trips')
plt.xticks(rotation=45)

# Add counts above the bars
for p in ax.patches:
    ax.annotate(format(int(p.get_height()), ',d'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 9), 
                textcoords='offset points')

plt.show()


# In[16]:


# Plot histogram of trips per pickup location
plt.figure(figsize=(15, 8))
ax = sns.countplot(y='pickup_location', data=df, order=df['pickup_location'].value_counts().index, palette='viridis')
plt.title('Number of Trips per Pickup Location')
plt.xlabel('Number of Trips')
plt.ylabel('Pickup Location')

# Add counts next to the bars
for p in ax.patches:
    ax.annotate(format(int(p.get_width()), ',d'), 
                (p.get_width(), p.get_y() + p.get_height() / 2.), 
                ha='center', va='center', 
                xytext=(10, 0), 
                textcoords='offset points')

plt.show()


# In[17]:


# Plot histogram of trips per dropoff location
plt.figure(figsize=(15, 8))
ax = sns.countplot(y='dropoff_location', data=df, order=df['dropoff_location'].value_counts().index, palette='viridis')
plt.title('Number of Trips per Dropoff Location')
plt.xlabel('Number of Trips')
plt.ylabel('Dropoff Location')

# Add counts next to the bars
for p in ax.patches:
    ax.annotate(format(int(p.get_width()), ',d'), 
                (p.get_width(), p.get_y() + p.get_height() / 2.), 
                ha='center', va='center', 
                xytext=(10, 0), 
                textcoords='offset points')

plt.show()


# In[18]:


# Calculate total distance and total duration of all trips
total_distance = df['trip_distance'].sum()  # Total distance in miles
total_duration_minutes = df['trip_duration'].sum()  # Total duration in minutes
total_duration_hours = total_duration_minutes / 60  # Convert total duration to hours

# Calculate average speed (miles per hour)
average_speed = total_distance / total_duration_hours

print("Total Distance (miles):", total_distance)
print("Total Duration (hours):", total_duration_hours)
print("Average Speed (miles per hour):", average_speed)


# In[19]:


# Create a new DataFrame with pickup and dropoff location names
pickup_dropoff_df = df[['pickup_location', 'dropoff_location']]

# Aggregate the number of trips for each unique combination of pickup and dropoff locations
heatmap_data = pickup_dropoff_df.groupby(['pickup_location', 'dropoff_location']).size().unstack(fill_value=0)

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap='viridis', cbar_kws={'label': 'Number of Trips'})
plt.title('High Demand Areas (Pickup and Dropoff) Heatmap')
plt.xlabel('Dropoff Location')
plt.ylabel('Pickup Location')
plt.show()


# In[20]:


# Extract the week number
df['week_number'] = df['date_time'].dt.week

# Group by customer_id and week_number, and count the number of trips
trips_per_week = df.groupby(['customer_id', 'week_number']).size().reset_index(name='num_trips')

# Calculate the average number of trips per week for each customer
avg_trips_per_customer = trips_per_week.groupby('customer_id')['num_trips'].mean()

# Select the top 10 customers with the highest average number of trips per week
top_10_customers_avg_trips = avg_trips_per_customer.nlargest(10)

# Filter trips_per_week DataFrame to include only data for top 10 customers
trips_per_week_top_10 = trips_per_week[trips_per_week['customer_id'].isin(top_10_customers_avg_trips.index)]

# Print the values of the average number of trips per week for each of the top 10 customers
print("\nAverage number of trips per week for each of the top 10 customers:")
print(avg_trips_per_customer.loc[top_10_customers_avg_trips.index])


# In[21]:


# Extract the month and month name
df['month'] = df['date_time'].dt.month
df['month_name'] = df['date_time'].dt.strftime('%B')  # Get the full name of the month

# Calculate the total spending for each customer for each month
df['total_spending'] = df['fare_amount']

# Group by customer_id, month, and month name, and sum the total spending
total_spending_per_month = df.groupby(['customer_id', 'month', 'month_name'])['total_spending'].sum().reset_index()

# Calculate the average spending per month
average_spending_per_month = total_spending_per_month.groupby(['month', 'month_name'])['total_spending'].mean()

# Print the average spending per month
print("Average spending of total customers per month:")
print(average_spending_per_month)


# In[22]:


# Calculate trip frequency for each customer
trip_frequency = df.groupby('customer_id').size().reset_index(name='trip_frequency')

# Calculate total spending for each customer
total_spending = df.groupby('customer_id')['fare_amount'].sum().reset_index(name='total_spending')

# Merge trip frequency and total spending dataframes
customer_data = pd.merge(trip_frequency, total_spending, on='customer_id')

# Normalize the data
scaler = StandardScaler()
customer_data_scaled = scaler.fit_transform(customer_data[['trip_frequency', 'total_spending']])

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
customer_data['cluster'] = kmeans.fit_predict(customer_data_scaled)

# Analyze the clusters
cluster_analysis = customer_data.groupby('cluster').mean()

# Print the cluster analysis
print("Cluster Analysis:")
print(cluster_analysis)

# Plot the graph
plt.figure(figsize=(10, 6))
plt.scatter(customer_data['trip_frequency'], customer_data['total_spending'], c=customer_data['cluster'], cmap='viridis', s=50, alpha=0.5)
plt.title('Customer Segmentation based on Trip Frequency and Total Spending')
plt.xlabel('Trip Frequency')
plt.ylabel('Total Spending')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()


# In[23]:


# Group by driver_id and week_number, and count the number of trips
trips_per_driver_per_week = df.groupby(['driver_id', 'week_number']).size().reset_index(name='num_trips')

# Count the number of unique drivers in each week
unique_drivers_per_week = df.groupby('week_number')['driver_id'].nunique().reset_index(name='num_drivers')

# Merge the two dataframes on week_number
merged_data = pd.merge(trips_per_driver_per_week, unique_drivers_per_week, on='week_number')

# Calculate the average trips per driver per week
merged_data['avg_trips_per_driver'] = merged_data['num_trips'] / merged_data['num_drivers']

# Print the average trips per driver per week
print("Average trips per driver per week:")
print(merged_data)


# In[24]:


# Group by driver_id and week_number, and count the number of trips
trips_per_driver_per_week = df.groupby(['driver_id', 'week_number']).size().reset_index(name='num_trips')

# Count the number of unique drivers in each week
unique_drivers_per_week = df.groupby('week_number')['driver_id'].nunique().reset_index(name='num_drivers')

# Merge the two dataframes on week_number
merged_data = pd.merge(trips_per_driver_per_week, unique_drivers_per_week, on='week_number')

# Calculate the average trips per driver per week
merged_data['avg_trips_per_driver'] = merged_data['num_trips'] / merged_data['num_drivers']

# Calculate the average trips per driver per week
avg_trips_per_driver_per_week = merged_data.groupby('driver_id')['avg_trips_per_driver'].mean().reset_index()

# Select the top 10 drivers based on average trips per week
top_10_drivers = avg_trips_per_driver_per_week.nlargest(10, 'avg_trips_per_driver')

# Filter the merged_data to include only data for top 10 drivers
merged_data_top_10 = merged_data[merged_data['driver_id'].isin(top_10_drivers['driver_id'])]

# Plot the average trips per week for top 10 drivers
plt.figure(figsize=(10, 6))
for driver_id, data in merged_data_top_10.groupby('driver_id'):
    plt.plot(data['week_number'], data['avg_trips_per_driver'], label=driver_id)
plt.title('Average Trips per Week for Top 10 Drivers')
plt.xlabel('Week Number')
plt.ylabel('Average Trips per Driver')
plt.legend(title='Driver ID', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()

# Print the average trips per driver per week for top 10 drivers
print("Average Trips per Driver per Week for Top 10 Drivers:")
print(top_10_drivers)


# In[25]:


# Calculate the total fare collected for each driver
total_fare_per_driver = df.groupby('driver_id')['fare_amount'].sum().reset_index(name='total_fare')

# Select the top 10 drivers based on total fare collected
top_10_drivers = total_fare_per_driver.nlargest(10, 'total_fare')

# Print the total fare collected by each of the top 10 drivers
print("Total Fare Collected by Top 10 Drivers:")
print(top_10_drivers)


# In[26]:


# Calculate total revenue from each payment method
total_revenue = df.groupby('payment_method')['fare_amount'].sum()

# Plot the pie chart
plt.figure(figsize=(8, 8))
plt.pie(total_revenue, labels=total_revenue.index, autopct='%1.1f%%', startangle=140)
plt.title('Comparison of Revenue from Payment Methods')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[27]:


# Sort the DataFrame by fare amount in descending order
top_10_fares = df.nlargest(10, 'fare_amount')

# Select the customer ID and fare amount columns
top_10_fares = top_10_fares[['customer_id', 'fare_amount']]

# Print the top 10 fare amounts with customer IDs
print("Top 10 Fare Amounts with Customer IDs:")
print(top_10_fares)


# In[28]:


# Calculate the total revenue
total_revenue = df['fare_amount'].sum()
# Print the total revenue
print("Total Revenue from the Whole Data: $", total_revenue)


# In[31]:


# Assuming 'time_of_day' is extracted from 'date_time'
df['time_of_day'] = df['date_time'].dt.hour

# Select relevant features and target variable
X = df[['trip_distance', 'time_of_day']]
y = df['trip_duration']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print("Root Mean Squared Error (RMSE):", rmse)


# In[32]:


y_actual= y_test

# Calculate RMSE for your model
rmse_model = np.sqrt(mean_squared_error(y_actual, y_pred))

# Define baseline model or industry standard
# For example, if the baseline is predicting the mean trip duration
baseline_prediction = np.mean(y_actual)

# Calculate RMSE for baseline model
rmse_baseline = np.sqrt(mean_squared_error(y_actual, np.full_like(y_actual, baseline_prediction)))

# Compare RMSE of your model with baseline
if rmse_model < rmse_baseline:
    print("Your model outperforms the baseline.")
elif rmse_model == rmse_baseline:
    print("Your model meets the baseline.")
else:
    print("Your model does not outperform the baseline.")

