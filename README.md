# Recommendation_Amazon_Products
E-commerce platforms like Amazon and Flipkart utilize advanced recommendation models to offer personalized product suggestions to their users. Amazon, for instance, uses item-to-item collaborative filtering, a technique designed to manage vast amounts of data while providing real-time, accurate recommendations. This approach works by comparing items a user has interacted with—whether through purchases or ratings—with similar items, and then curating a customized list of recommendations. In this project, our goal is to develop a recommendation system specifically for Amazon's electronics products.

## Libraries
```
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from surprise import SVDpp, Dataset, Reader,SVD
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from surprise import accuracy
from surprise.model_selection import GridSearchCV
from surprise.model_selection import train_test_split, cross_validate
```

## Checking The Data
We import the data set and gave the columns names
```
data = pd.read_csv("C:/Users/kaanc/Python_tenserflow/ratings_Electronics.csv",names=['userId', 'productId','Rating','timestamp'])
```

userId: A unique identifier assigned to each user.                   
productId: A unique identifier assigned to each product.                      
Rating: The rating given by the user to the product (on a scale of 1 to 5).                   
timestamp: The time when the rating was given, in Unix timestamp format.              

```
data.head(10)
```
![image](https://github.com/user-attachments/assets/0c98fcb8-f9d7-496b-bd2e-852a62a4973b)

then we checked the columns and types.
```
data.info()
```
![image](https://github.com/user-attachments/assets/a8c49532-bc3f-4416-9db8-3a17515f3592)

```
data.describe()["Rating"].T
```
![image](https://github.com/user-attachments/assets/cecefa30-7b49-4c1c-82c8-257f664f7655)

```
print("Minimum Rating: %d"%(min(data["Rating"])))
print("Maximum Rating: %d"%(max(data["Rating"])))
```
![image](https://github.com/user-attachments/assets/dd66133a-c419-43d4-b616-5fafac36ffa1)

## Mising Values Check
```
data.isnull().sum()
data.drop_duplicates(inplace=True)
```
We don't have missing values.

## EDA 
```
plt.figure(figsize=(10, 6))
sns.histplot(data['Rating'], bins=5, kde=False, color='skyblue', edgecolor='black')
plt.title('Distribution of Ratings', fontsize=16)
plt.xlabel('Rating', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
```
![image](https://github.com/user-attachments/assets/c27285b5-fb2e-4515-bb2d-845cf85c406f)

This histogram illustrates the distribution of ratings given by users. The most common rating is 5 stars, indicating that a majority of users are satisfied with the products. The frequent occurrence of 1-star ratings also reveals that some products are strongly disliked. The ratings are generally concentrated at the extremes, with fewer moderate ratings. This suggests that users tend to have strong opinions about the products, and recommendation systems should consider this tendency. Overall, the distribution of ratings provides significant insights into product quality and user satisfaction.

```
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
ratings_over_time = data.groupby(data['timestamp'].dt.to_period("M")).size()

plt.figure(figsize=(12, 6))
ratings_over_time.plot()
plt.title('Number of Ratings Over Time', fontsize=16)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Number of Ratings', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
```
![image](https://github.com/user-attachments/assets/1749257d-97fc-4f40-8cfc-029bb3718cd8)

This graph shows the change in the number of ratings given by users from 1999 to 2014. In the early 2000s, rating activity was quite low, but there is a noticeable increase around 2007. This increase may be related to the rapid rise in popularity of e-commerce sites like Amazon and the growing trust of users in online shopping. After 2011, there is a sharp rise in the number of ratings, which can be explained by the widespread adoption of e-commerce and more users joining these platforms. In 2013, rating activity reaches its peak, followed by a slight decline. This decline could be due to seasonal effects or changes in user habits. Overall, the graph clearly illustrates the expansion of e-commerce and how user product rating habits have evolved over time.

```
print("No Ranking Values :",data.shape[0])
print("Total No of Users   :", len(np.unique(data.userId)))
print("Total No of products  :", len(np.unique(data.productId)))
```
![image](https://github.com/user-attachments/assets/4b2d8f4a-3f89-4c96-b410-08f19a48f33a)

```
user_activity = data.groupby('userId').size()
active_users = user_activity[user_activity >= 10].index

product_popularity = data.groupby('productId').size()
popular_products = product_popularity[product_popularity >= 10].index

filtered_data = data[data['userId'].isin(active_users) & data['productId'].isin(popular_products)]
```

```
active_users = data.groupby('userId')['Rating'].count().nlargest(10)
print(active_users)
```
![Screenshot 2024-08-23 200221](https://github.com/user-attachments/assets/859c013b-11f3-4c34-94e0-dfebe86c3c41)

This table lists the most active users in the dataset.

```
popular_products = data.groupby('productId')['Rating'].count().nlargest(10)
print(popular_products)
```
![image](https://github.com/user-attachments/assets/989c52d8-bf24-4dca-a9d2-55162a9cdf59)

This table shows the most popular products in the dataset. Overall, this list indicates which products are the most popular among users and have received the most feedback.

```
data['Year'] = data['timestamp'].dt.year
data['Month'] = data['timestamp'].dt.month

heatmap_data = data.pivot_table(index='Year', columns='Month', values='Rating', aggfunc='count')

plt.figure(figsize=(12, 5))
sns.heatmap(heatmap_data, cmap='crest', annot=True, fmt='g')
plt.title('Heatmap of Reviews Over Time')
plt.xlabel('Month')
plt.ylabel('Year')
plt.show()
```
![image](https://github.com/user-attachments/assets/5ac9f810-2659-4e7b-ad70-3022d9772d5a)

This heatmap shows the distribution of user reviews by month from 1999 to 2014. In the early years (1999-2006), the number of reviews was quite low, but a significant increase is observed starting in 2007. This rise can be attributed to the rapid growth of e-commerce platforms and the increasing trust of users in these platforms. Between 2011 and 2013, especially in May and June, there are peak points in the number of reviews, which could be explained by major sales campaigns or special events. In 2013 and 2014, the number of reviews reached its highest levels on both an annual and monthly basis, with the first half of 2014 showing particularly high numbers. This indicates that e-commerce had matured and users were providing feedback more intensely. Additionally, there is a general upward trend in the number of reviews in the last quarter of each year, likely related to the high volume of holiday season shopping.

## Cleaning The data 
We remove the timestamp cause we don't need it.
```
data.drop("timestamp",axis=1,inplace=True)
```

## Modelling 
This code takes a sample of 50,000 rows from the dataset, converts this sample into a dataset usable by the Surprise library, and creates a simplified subset of the dataset containing user-product-rating information.
```
data = data[['userId', 'productId', 'Rating']]
reader = Reader(rating_scale=(1, 5))
data_sample = data.sample(n=50000, random_state=42)
surprise_data = Dataset.load_from_df(data_sample[['userId', 'productId', 'Rating']], reader)
```

This function calculates the RMSE and MAE metrics to evaluate the model's performance and returns the results as a dictionary.
```
def calculate_metrics(predictions):
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    
    return {"RMSE": rmse, "MAE": mae}
```

This code uses GridSearchCV to find the best parameters for the SVD++ model by optimizing its hyperparameters, and then retrains the model with these parameters. Afterwards, it evaluates the model's performance using RMSE and MAE metrics and prints the results.
```
param_grid = {
    'n_factors': [20, 50, 100, 150],  
    'n_epochs': [10, 20, 30, 50],    
    'lr_all': [0.002, 0.005, 0.01],  
    'reg_all': [0.01, 0.02, 0.1, 0.2]  
}
gs = GridSearchCV(SVDpp, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1)
gs.fit(surprise_data)

best_params = gs.best_params['rmse']
print("Best parameters:", best_params)

best_model = SVDpp(**best_params)
trainset = surprise_data.build_full_trainset()
best_model.fit(trainset)

testset = trainset.build_testset()
predictions = best_model.test(testset)
metrics = calculate_metrics(predictions)
print("Hyperparameter-tuned model performance:", metrics)
```
![image](https://github.com/user-attachments/assets/49bb7231-4a68-4f7e-a0d0-c8b02f35462c)

This code makes predictions for products that have not yet been rated by a specific userId, sorts these predictions, and recommends the top 10 highest-rated products to the user. As a result, a list of products that the user might be interested in is provided.
```
user_id_input = input("Please enter a userId: ")
all_product_ids = data['productId'].unique()
user_unrated_products = [(user_id_input, product_id, 0) for product_id in all_product_ids]
predictions = [best_model.predict(uid=user_id_input, iid=product_id) for user_id_input, product_id, _ in user_unrated_products]
predicted_ratings = pd.DataFrame([(pred.uid, pred.iid, pred.est) for pred in predictions],
                                 columns=['userId', 'productId', 'predicted_rating'])
top_predictions = predicted_ratings.sort_values(by='predicted_rating', ascending=False).head(10)
print("Recommnded products for user {}:".format(user_id_input))
print(top_predictions[['productId', 'predicted_rating']])
```
![image](https://github.com/user-attachments/assets/13b0510e-44d3-4de7-bad5-c990c58154ef)

This output shows a list of recommended products and the predicted ratings for the user with the ID A2AY4YUOX2N1BQ. The model has predicted a rating of 4.95 for the product B001W28L2Y, and ratings of 4.93 for the products B004M5H660 and B003ES5ZUU. The predicted ratings for the other products range from 4.83 to 4.99. These high ratings suggest that the user is likely to be highly satisfied with these products and that they might be of interest to the user. The model's predictions have been used to identify products that align with the user's preferences and that they are likely to enjoy.
