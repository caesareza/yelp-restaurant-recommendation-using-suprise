import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px
from textblob import TextBlob
# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
sns.set(style="whitegrid", context='talk')

businesses = pd.read_csv('../yelp/yelp_academic_dataset_business.csv', nrows=50000)
reviews = pd.read_csv('../yelp/yelp_academic_dataset_review.csv', nrows=50000)

restoran = businesses[['business_id','name','address', 'categories', 'attributes','stars']]
print('data awal', restoran.shape)
mask_restaurants = restoran['categories'].str.contains('Restaurants')
# create a mask for food
mask_food = restoran['categories'].str.contains('Food')
# apply both masks
restaurants_and_food = restoran[mask_restaurants & mask_food]
# number of businesses that have food and restaurant in their category
restaurants_and_food.drop_duplicates(subset='name', keep=False, inplace=True)
review = reviews[['review_id','business_id','user_id','text']]
combined_business_data = pd.merge(restaurants_and_food, review, on='business_id')
print(combined_business_data.head(10))

combined_business_data.rename(columns = {'stars':'rating'}, inplace = True)
data = combined_business_data[['user_id', 'business_id', 'name', 'text', 'rating']]
# map floating point stars to an integer
mapper = {1.0:1,1.5:2, 2.0:2, 2.5:3, 3.0:3, 3.5:4, 4.0:4, 4.5:5, 5.0:5}
data['rating'] = data['rating'].map(mapper)
data = data[data['rating'] != 3]
data['sentiment'] = data['rating'].apply(lambda rating: +1 if rating > 3 else -1)

# split df - positive and negative sentiment:
positive = data[data['sentiment'] == +1]
negative = data[data['sentiment'] == -1]

print(positive.head(10))
print(negative.head(10))
print(data.head(10))
print(data.shape)

data['sentimentt'] = data['sentiment'].replace({-1 : 'negative'})
data['sentimentt'] = data['sentimentt'].replace({1 : 'positive'})
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
print(data.nlargest(10, ['polarity']))
print(data.nsmallest(5, ['polarity']))
data = data.nlargest(10, ['polarity'])

for var in ['polarity', 'subjectivity']:
    plt.figure(figsize=(12,4))
    sns.histplot(data.query("sentiment==1")[var], bins=30, kde=False,
                 color='green', label='Positive')
    sns.histplot(data.query("sentiment==-1")[var], bins=30, kde=False,
                 color='red', label='Negative')
    plt.legend()
    plt.title(f'Histogram of {var} by true sentiment');
    plt.show()

print(data.head(10))

from surprise import KNNBaseline
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
reader = Reader()
data = Dataset.load_from_df(data[['user_id', 'business_id', 'rating']], reader)
trainset = data.build_full_trainset()
sim_options = {'name': 'pearson_baseline', 'user_based': False}
algo = KNNBaseline(k=10, sim_options=sim_options)
algo.fit(trainset)

accuracy.rmse(predictions)

