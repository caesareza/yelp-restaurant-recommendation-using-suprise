import pandas as pd

businesses = pd.read_csv('../yelp/yelp_academic_dataset_business.csv', nrows=100000)
reviews = pd.read_csv('../yelp/yelp_academic_dataset_review.csv', nrows=100000)
# print(reviews.head(10))

restoran = businesses[['business_id','name','address', 'categories', 'attributes','stars']]
mask_restaurants = restoran['categories'].str.contains('Restaurants')
# create a mask for food
mask_food = restoran['categories'].str.contains('Food')
# apply both masks
restaurants_and_food = restoran[mask_restaurants & mask_food]
# number of businesses that have food and restaurant in their category
restaurants_and_food.drop_duplicates(subset='name', keep=False, inplace=True)


review = reviews[['review_id','business_id','user_id']]
combined_business_data = pd.merge(restaurants_and_food, review, on='business_id')
# print(combined_business_data.shape)
print(combined_business_data[['name', 'categories', 'user_id', 'business_id']].head(10))

from surprise import Reader, Dataset, SVD
from surprise.model_selection.validation import cross_validate
reader = Reader()

data = Dataset.load_from_df(combined_business_data[['user_id', 'business_id', 'stars']], reader)

# create a user-item matrix
rating_crosstab = combined_business_data.pivot_table(values='stars', index='user_id', columns='name', fill_value=0)
print(rating_crosstab.head(10))
print(rating_crosstab.shape)

# Transpose the Utility matrix
X = rating_crosstab.values.T
X.shape
print(X.shape)

import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score
import numpy as np

SVD = TruncatedSVD(n_components=12, random_state=17)
result_matrix = SVD.fit_transform(X)
result_matrix.shape
print(result_matrix.shape)


# PearsonR coef
corr_matrix = np.corrcoef(result_matrix)
corr_matrix.shape
print(corr_matrix.shape)

restaurant_names = rating_crosstab.columns
restaurants_list = list(restaurant_names)

popular_rest = restaurants_list.index('Banzai Sushi')
print("index of the popular restaurant: ", popular_rest)

# restaurant of interest
corr_popular_rest = corr_matrix[popular_rest]
corr_popular_rest.shape
print(corr_popular_rest.shape)
print(list(restaurant_names[(corr_popular_rest < 1.0) & (corr_popular_rest > 0.9)]))

# # svd = SVD()
# #
# # # Run 5-fold cross-validation and print results
# # cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
# #
# # trainset = data.build_full_trainset()
# # svd.fit(trainset)
# #
#
# trainsetfull = data.build_full_trainset()
# print('Number of users: ', trainsetfull.n_users, '\n')
# print('Number of items: ', trainsetfull.n_items, '\n')
# print('Number of items: ', trainsetfull.rating_scale, '\n')
#
# from surprise import KNNWithMeans
# from surprise import accuracy
#
# my_k = 15
# my_min_k = 5
# my_sim_option = {
#     'name': 'pearson', 'user_based': False
# }
#
# algo = KNNWithMeans(
#     k=my_k, min_k=my_min_k,
#     sim_options=my_sim_option, verbose=True
# )
#
# results = cross_validate(
#     algo=algo, data=data, measures=['RMSE'],
#     cv=5, return_train_measures=True
# )
#
# print(results['test_rmse'].mean())
#
# # Step 3 - Model Fitting
#
# algo.fit(trainsetfull)
#
# # Step 4 - Prediction
# # rVBPQdeayMYht4Uv_FOLHg
#
#
# r = restaurants_and_food.copy()
# r['Estimate_Score'] = r['business_id'].apply(lambda x: algo.predict('0e8g0QxR2JFFfne6vpUoGQ', x).est)
#
#
# r = r.sort_values(by=['Estimate_Score'], ascending=False)
# print(r[['name', 'categories', 'stars', 'Estimate_Score']].head(10))