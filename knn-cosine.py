import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px
from textblob import TextBlob
# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", context='talk')

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
review = reviews[['review_id','business_id','user_id','text']]
combined_business_data = pd.merge(restaurants_and_food, review, on='business_id')
# print(reviews.columns)

from surprise import Reader, Dataset, KNNWithMeans
from surprise.model_selection.validation import cross_validate
reader = Reader()
combined_business_data.rename(columns = {'stars':'rating'}, inplace = True)
data = combined_business_data[['user_id', 'business_id', 'text', 'rating']];
# map floating point stars to an integer
mapper = {1.0:1,1.5:2, 2.0:2, 2.5:3, 3.0:3, 3.5:4, 4.0:4, 4.5:5, 5.0:5}
data['rating'] = data['rating'].map(mapper)
data = data[data['rating'] != 3]
data['sentiment'] = data['rating'].apply(lambda rating: +1 if rating > 3 else -1)


# split df - positive and negative sentiment:
positive = data[data['sentiment'] == +1]
negative = data[data['sentiment'] == -1]

# print(positive.head(10))
# print(negative.head(10))

# data['sentimentt'] = data['sentiment'].replace({-1 : 'negative'})
# data['sentimentt'] = data['sentimentt'].replace({1 : 'positive'})
# fig = px.histogram(data, x="sentimentt")
# fig.update_traces(marker_color="indianred",marker_line_color='rgb(8,48,107)',
#                   marker_line_width=1.5)
# fig.update_layout(title_text='Product Sentiment')
# fig.show()

def remove_punctation(text):
    final = "".join(u for u in text if u not in ("?", ".", ";", ":", "(", ")", "!",'"'))
    return final

data['text'] = data['text'].apply(remove_punctation)
data[['polarity', 'subjectivity']] = data['text'].apply(lambda x:TextBlob(x).sentiment).to_list()
# print(data.nlargest(5, ['polarity']))
# print(data.nsmallest(5, ['polarity']))
data = data.nlargest(500, ['polarity'])

for var in ['polarity', 'subjectivity']:
    plt.figure(figsize=(12,4))
    sns.histplot(data.query("sentiment==1")[var], bins=30, kde=False,
                 color='green', label='Positive')
    sns.histplot(data.query("sentiment==-1")[var], bins=30, kde=False,
                 color='red', label='Negative')
    plt.legend()
    plt.title(f'Histogram of {var} by true sentiment');
    # plt.show()



data = Dataset.load_from_df(data[['user_id', 'business_id', 'rating']], reader)

from collections import defaultdict
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import KFold
from surprise import KNNBasic, KNNBaseline

from surprise.model_selection.split import train_test_split
trainset, testset = train_test_split(data, test_size=0.33)

sim_options = {'name': 'cosine',
               'user_based': True
               }
algo = KNNBaseline(k=5, sim_options=sim_options)
dataUnique = combined_business_data.drop_duplicates(subset="name")
predictions = algo.fit(trainset).test(testset)
accuracy.rmse(predictions)
result = pd.DataFrame(predictions)
result.drop(columns = {'details'}, inplace = True)
# predictions = predictions.nlargest(3,  ['r_ui'])
print(result.head())
tampil = pd.Series(dataUnique.index)
# for r in result:
#     print(r[r['uid']])

# result = result.join(dataUnique,on='iid')

def get_top_n(predictions, restaurant):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = []
    i = 10
    for uid, iid, true_r, est, _ in predictions:
        top_n.append(dataUnique[dataUnique['business_id'] == iid])
    print(top_n)

get_top_n(predictions, dataUnique)


