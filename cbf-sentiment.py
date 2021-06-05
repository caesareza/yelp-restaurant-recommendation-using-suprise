import numpy as np
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

businesses = pd.read_csv('../yelp/yelp_academic_dataset_business.csv', nrows=5000)
reviews = pd.read_csv('../yelp/yelp_academic_dataset_review.csv', nrows=5000)

restoran = businesses[['business_id','name','address', 'categories', 'attributes','stars']]
mask_restaurants = restoran['categories'].str.contains('Restaurants')
# create a mask for food
mask_food = restoran['categories'].str.contains('Food')
# apply both masks
restaurants_and_food = restoran[mask_restaurants & mask_food]
# number of businesses that have food and restaurant in their category
restaurants_and_food.drop_duplicates(subset='name', keep=False, inplace=True)
review = reviews[['review_id','business_id','user_id','text']]
data = pd.merge(restaurants_and_food, review, on='business_id')
data.rename(columns = {'stars':'rating'}, inplace = True)

df_categories_dummies = pd.Series(data['categories']).str.get_dummies(',')
print(df_categories_dummies.shape)
print(df_categories_dummies.columns)

# data = restaurants_and_food
data = data[['user_id','business_id','name','text','rating']];
# map floating point stars to an integer
mapper = {1.0:1,1.5:2, 2.0:2, 2.5:3, 3.0:3, 3.5:4, 4.0:4, 4.5:5, 5.0:5}
data['rating'] = data['rating'].map(mapper)
data = data[data['rating'] != 3]
data['sentiment'] = data['rating'].apply(lambda rating: +1 if rating > 3 else 0)

# split df - positive and negative sentiment:
positive = data[data['sentiment'] == +1]
negative = data[data['sentiment'] == 0]

# # print(positive.head(10))
# # print(negative.head(10))
#
# # data['sentimentt'] = data['sentiment'].replace({-1 : 'negative'})
# # data['sentimentt'] = data['sentimentt'].replace({1 : 'positive'})
# # fig = px.histogram(data, x="sentimentt")
# # fig.update_traces(marker_color="indianred",marker_line_color='rgb(8,48,107)',
# #                   marker_line_width=1.5)
# # fig.update_layout(title_text='Product Sentiment')
# # fig.show()

def remove_punctation(text):
    final = "".join(u for u in text if u not in ("?", ".", ";", ":", "(", ")", "!",'"'))
    return final

data['text'] = data['text'].apply(remove_punctation)
data[['polarity', 'subjectivity']] = data['text'].apply(lambda x:TextBlob(x).sentiment).to_list()
# print(data.nlargest(5, ['polarity']))
# print(data.nsmallest(5, ['polarity']))
# data = data.nlargest(10, ['polarity'])

for var in ['polarity', 'subjectivity']:
    plt.figure(figsize=(12,4))
    sns.histplot(data.query("sentiment==1")[var], bins=30, kde=False,
                 color='green', label='Positive')
    sns.histplot(data.query("sentiment==-1")[var], bins=30, kde=False,
                 color='red', label='Negative')
    plt.legend()
    plt.title(f'Histogram of {var} by true sentiment');
    # plt.show()


# data['polarity'] = data['polarity'].apply(lambda polarity: +1 if rating > 3 else -1)
data['blob_polarity'] = np.where(data['polarity']>0, 1, 0)
# Concat all tables and drop Restaurant column
df_final = pd.concat([df_categories_dummies, data], axis=1)
df_final = df_final.dropna()
df_final['blob_polarity'] = df_final['blob_polarity'].astype(int)
# print(df_final.head(10))

# Set target
target = 'rating'
features = df_final.drop(columns=['user_id','business_id','name','text','sentiment','polarity','subjectivity','rating']).columns

# Create X (all the features) and y (target)
X = df_final[features]
print(X)
y = df_final[target]
print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=.2, random_state=8,
                                                    stratify=data[target])

# Inspect data
print(f"Training data ({X_train.shape[0]} rows): Target distribution")
print(y_train.value_counts(normalize=True))
print(f"\nTest data ({X_test.shape[0]} rows): Target distribution")
print(y_train.value_counts(normalize=True))

# Define feature groups
numerical = X_train.select_dtypes(['number']).columns
print(f'\nNumerical: {numerical}')
categorical = X_train.columns.difference(numerical)
X_train[categorical] = X_train[categorical].astype('object')
print(f'Categorical: {categorical}')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors=22)
knn.fit(X_train, y_train)

print('X_test_knn shape',X_train.shape)
print('y_train_knn shape',y_train.shape)

accuracy_train = knn.score(X_train, y_train)
accuracy_test = knn.score(X_test, y_test)


from sklearn import neighbors
from math import sqrt
from sklearn.metrics import mean_squared_error
rmse_val = [] #to store rmse values for different k
kvalue = [10, 20, 30, 50, 80]
for K in kvalue:
    model = neighbors.KNeighborsRegressor(n_neighbors=K)
    model.fit(X_train, y_train)  # fit the model
    pred = model.predict(X_test)  # make prediction on test set
    error = sqrt(mean_squared_error(y_test, pred))  # calculate rmse
    rmse_val.append(error)  # store rmse values
    print('RMSE value for k= ', K, 'is:', error)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import KFold
from numpy import mean
from numpy import std
# prepare the cross-validation procedure
cv = KFold(n_splits=5, random_state=1, shuffle=True)
# create model
model = LogisticRegression()
cvs = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
y_pred = cross_val_predict(model, X_train, y_train, cv=cv)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
mae = mean_absolute_error(y_train, y_pred)
mse = mean_squared_error(y_train, y_pred)
rmse = mean_squared_error(y_train, y_pred, squared=False)
# residuals = y_test - y_pred

print('y_test',y_test)
print('y_pred',y_pred)
print(f"Score on training set: {accuracy_train}")
print(f"Score on test set: {accuracy_test}")

# report performance
print('Accuracy: %.3f (%.3f)' % (mean(cvs), std(cvs)))
print('MAE', mae)
print('MSE', mse)
print('RMSE', rmse)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_pred)
print('confusion matrix', cm)

# Recall
from sklearn.metrics import recall_score
recall = recall_score(y_train, y_pred, average='weighted')
print('recall:', recall)

# Precision
from sklearn.metrics import precision_score
precision = precision_score(y_train, y_pred, average='weighted')
print('precision:', precision)

# f1_score
from sklearn.metrics import f1_score
f1_score(y_train, y_pred, average=None)
# Method 2: Manual Calculation
F1 = 2 * (precision * recall) / (precision + recall)
print('F-Measure:', F1)

# look at the last row for the test
final = df_final.drop(['user_id', 'business_id', 'text', 'name'], axis = 1)
# look at the last row for the test
print(final.iloc[-1:])

# look at the restaurant name from the last row.
print("Validation set (Restaurant name): ", df_final.values[-1])


# test set from the df_final table (only last row)
test_set = df_final.iloc[-1:,:-2]

# test set from the df_final table (only last row)
test_set = final.iloc[-1:,:-2]

# validation set from the df_final table (exclude the last row)
X_val =  final.iloc[:-1,:-2]
y_val = final['rating'].iloc[:-1]

# fit model with validation set
n_knn = knn.fit(X_val, y_val)

# distances and indeces from validation set
distances, indeces =  n_knn.kneighbors(test_set)
#n_knn.kneighbors(test_set)[1][0]

# create table distances and indeces
final_table = pd.DataFrame(n_knn.kneighbors(test_set)[0][0], columns = ['distance'])
final_table['index'] = n_knn.kneighbors(test_set)[1][0]
final_table.set_index('index')

print(final_table.set_index('index'))

# get names of the restaurant that similar to the last row
result = final_table.join(df_final,on='index')
hasil = result.head(10)
hasil = hasil.dropna()
hasil.drop_duplicates(subset='name', keep=False, inplace=True)
print(hasil)