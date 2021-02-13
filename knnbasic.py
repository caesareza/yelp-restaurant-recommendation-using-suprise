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
# print(restaurants_and_food.head(50))

review = reviews[['review_id','business_id','user_id']]
combined_business_data = pd.merge(restaurants_and_food, review, on='business_id')
print(combined_business_data.head(5))
print(combined_business_data.shape)

from surprise import Reader, Dataset, KNNWithMeans

from surprise.model_selection.validation import cross_validate
reader = Reader()
data = Dataset.load_from_df(combined_business_data[['user_id', 'business_id', 'stars']], reader)

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for user_id, _, true_r, est, _ in predictions:
        user_est_true[user_id].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for user_id, stars in user_est_true.items():

        # Sort user ratings by estimated value
        stars.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in stars)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in stars[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in stars[:k])

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[user_id] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[user_id] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls


# from surprise.model_selection import train_test_split
# trainset, testset = train_test_split(data, test_size=.25)
from collections import defaultdict
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import KFold
from surprise import KNNWithMeans

# We'll use the famous SVD algorithm.
kf = KFold(n_splits=5)
sim_options = {'name': 'cosine', 'user_based': False}
algo = KNNWithMeans(sim_options=sim_options)
# algo = SVD()

def get_top_n(predictions, n=10):
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
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


for trainset, testset in kf.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)
    accuracy.rmse(predictions)
    accuracy.mae(predictions)

    top_n = get_top_n(predictions, n=5)

    precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=4)

    # Precision and recall can then be averaged over all users
    print('precision', sum(prec for prec in precisions.values()) / len(precisions))
    print('recall', sum(rec for rec in recalls.values()) / len(recalls))

# Retrieve inner id of the movie Toy Story
# trainset = data.build_full_trainset()
# toy_story_raw_id = 'eSQ3z93DlzkpXK_H6MFEMw'
# toy_story_inner_id = algo.trainset.to_inner_uid(toy_story_raw_id)

# Retrieve inner ids of the nearest neighbors of Toy Story.


r = restaurants_and_food.copy()
r['Estimate_Score'] = r['business_id'].apply(lambda x: algo.predict('gRdBkmXdRqUzDMkcMtt7rQ', x).est)

r = r.sort_values(by=['Estimate_Score'], ascending=False)
print(r[['business_id', 'name', 'categories', 'stars', 'Estimate_Score']].head(10))
