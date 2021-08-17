import numpy as np
import pandas as pd
from textblob import TextBlob

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

review = reviews[['review_id','business_id','user_id', 'text']]
combined_business_data = pd.merge(restaurants_and_food, review, on='business_id')
combined_business_data.rename(columns = {'stars':'rating'}, inplace = True)
data = combined_business_data[['user_id', 'business_id', 'name', 'text', 'rating']]
mapper = {1.0:1,1.5:2, 2.0:2, 2.5:3, 3.0:3, 3.5:4, 4.0:4, 4.5:5, 5.0:5}
data['rating'] = data['rating'].map(mapper)
data['sentiment'] = data['rating'].apply(lambda rating: +1 if rating > 3 else 0)

data['sentimentt'] = data['sentiment'].replace({0 : 'negative'})
data['sentimentt'] = data['sentimentt'].replace({1 : 'positive'})
print(data.head())

#split train test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['sentiment'], test_size=0.2, random_state=1, stratify=data['sentimentt'])

# Append sentiment back using indices
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

# Check dimensions
print(f"Train: {train.shape[0]} rows and {train.shape[1]} columns")
print(f"{train['sentiment'].value_counts()}\n")
print(f"Test: {test.shape[0]} rows and {test.shape[1]} columns")
print(test['sentiment'].value_counts())

print(train.head())
# print(data['sentimentt'].value_counts())

# data = data[data['rating'] != 3]
# print(data.head(10))
# print(data['sentimentt'].value_counts())

def remove_punctation(text):
  final = "".join(u for u in text if u not in ("?", ".", ";", ":", "(", ")", "!", '"'))
  return final

train['text'] = train['text'].apply(remove_punctation)
train[['polarity', 'subjectivity']] = train['text'].apply(lambda x:TextBlob(x).sentiment).to_list()

print(train.head())

from sklearn.metrics import classification_report, mean_squared_error
train['blob_polarity'] = np.where(train['polarity']>0, 1, 0)
target_names=['negative', 'positive']
print(classification_report(train['sentiment'],
                            train['blob_polarity'],
                            target_names=target_names))
print('RMSE = ',mean_squared_error(train['sentiment'], train['blob_polarity']))

train = train[train['polarity'] > 0]
print(train.head())

from surprise import Dataset
from surprise import Reader
reader = Reader()
train = Dataset.load_from_df(train[['text', 'sentiment', 'blob_polarity']], reader)

from surprise import SVD
from surprise.model_selection import cross_validate

svd = SVD(verbose=True, n_epochs=10)
cross_validate(svd, train, measures=['RMSE', 'MAE'], cv=3, verbose=True)
trainset = train.build_full_trainset()
svd.fit(trainset)
