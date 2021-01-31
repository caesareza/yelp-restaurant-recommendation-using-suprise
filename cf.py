import pandas as pd

businesses = pd.read_csv('../yelp/yelp_academic_dataset_business.csv', nrows=100000)
reviews = pd.read_csv('../yelp/yelp_academic_dataset_review.csv', nrows=100000)

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
print(combined_business_data.shape)
print(combined_business_data[['name', 'categories', 'user_id']].head(50))

from surprise import Reader, Dataset, SVD
from surprise.model_selection.validation import cross_validate
reader = Reader()

data = Dataset.load_from_df(combined_business_data[['user_id', 'business_id', 'stars']], reader)
svd = SVD()

# Run 5-fold cross-validation and print results
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

trainset = data.build_full_trainset()
svd.fit(trainset)













# r = restaurants_and_food.copy()
# r['Estimate_Score'] = r['business_id'].apply(lambda x: svd.predict('xIm6CP6pAqS3XQ7QF3Z89g', x).est)
#
#
# r = r.sort_values(by=['Estimate_Score'], ascending=False)
# print(r[['name', 'categories', 'stars', 'Estimate_Score']].head(10))
