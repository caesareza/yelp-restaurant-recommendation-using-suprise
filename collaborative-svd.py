import numpy as np
import pandas as pd

businesses = pd.read_csv('../yelp/yelp_academic_dataset_business.csv', nrows=10000)
reviews = pd.read_csv('../yelp/yelp_academic_dataset_review.csv', nrows=10000)

restoran = businesses[['business_id','name','address', 'categories', 'attributes','stars']]
mask_restaurants = restoran['categories'].str.contains('Restaurants')
# create a mask for food
mask_food = restoran['categories'].str.contains('Food')
# apply both masks
restaurants_and_food = restoran[mask_restaurants & mask_food]
# number of businesses that have food and restaurant in their category
restaurants_and_food.drop_duplicates(subset='name', keep=False, inplace=True)
# print(restaurants_and_food.head(50))

review = reviews[['review_id','business_id','user_id']]
combined_business_data = pd.merge(restaurants_and_food, review, on='business_id')
print(combined_business_data.head(5))
print(combined_business_data.shape)
print(reviews.columns)

from surprise import Dataset
from surprise import Reader
reader = Reader()
data = Dataset.load_from_df(reviews[['user_id', 'business_id', 'stars']], reader)

from surprise import SVD
from surprise.model_selection import cross_validate
svd = SVD(verbose=True, n_epochs=10)
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

trainset = data.build_full_trainset()
svd.fit(trainset)

print(svd.predict(uid='eSQ3z93DlzkpXK_H6MFEMw', iid='pQeaRpvuhoEqudo3uymHIQ'))

import difflib
import random


def get_book_id(restaurant_name, metadata):
    """
    Gets the restaurant ID for a book title based on the closest match in the metadata dataframe.
    """
    existing_titles = list(metadata['name'].values)
    closest_titles = difflib.get_close_matches(restaurant_name, existing_titles)
    business_id = metadata[metadata['name'] == closest_titles[0]]['name'].values[0]
    return business_id


def get_book_info(business_id, metadata):
    """
    Returns some basic information about a book given the book id and the metadata dataframe.
    """

    restaurant_info = metadata[metadata['business_id'] == business_id][['business_id', 'name', 'categories']]
    return restaurant_info.to_dict(orient='records')

def predict_review(user_id, business_name, model, metadata):
    """
    Predicts the review (on a scale of 1-5) that a user would assign to a specific book.
    """

    business_id = get_book_id(business_name, metadata)
    review_prediction = model.predict(uid=user_id, iid=business_id)
    return review_prediction.est


def generate_recommendation(user_id, model, metadata, thresh=4):
    """
    Generates a book recommendation for a user based on a rating threshold. Only
    books with a predicted rating at or above the threshold will be recommended
    """

    restaurant_names = list(metadata['name'].values)
    random.shuffle(restaurant_names)

    for restaurant_name in restaurant_names:
        rating = predict_review(user_id, restaurant_name, model, metadata)
        print(rating )
        # if rating >= thresh:
        #     business_id = get_book_id(restaurant_name, metadata)
        #     return get_book_info(business_id, metadata)

print(generate_recommendation('eSQ3z93DlzkpXK_H6MFEMw', svd, businesses))