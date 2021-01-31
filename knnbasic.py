import pandas as pd

businesses = pd.read_csv('../yelp/yelp_academic_dataset_business.csv', nrows=50000)
reviews = pd.read_csv('../yelp/yelp_academic_dataset_review.csv', nrows=50000)

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
print(combined_business_data.shape)

from surprise import Reader, Dataset, KNNWithMeans

from surprise.model_selection.validation import cross_validate
reader = Reader()
data = Dataset.load_from_df(combined_business_data[['user_id', 'business_id', 'stars']], reader)



# Train the algorithm on the trainset, and predict ratings for the testset
# algo.fit(trainset)
# predictions = algo.test(testset)
#
#
# # Then compute RMSE
# accuracy.rmse(predictions)
# accuracy.mae(predictions)

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls


# from surprise.model_selection import train_test_split
# trainset, testset = train_test_split(data, test_size=.25)
from collections import defaultdict
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import KFold
from surprise import KNNBaseline

# We'll use the famous SVD algorithm.
kf = KFold(n_splits=5)
sim_options = {'name': 'pearson_baseline', 'user_based': False}
algo = KNNBaseline(sim_options=sim_options)
# algo = SVD()

for trainset, testset in kf.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)
    accuracy.rmse(predictions)
    accuracy.mae(predictions)

    precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=4)

    # Precision and recall can then be averaged over all users
    print('precision', sum(prec for prec in precisions.values()) / len(precisions))
    print('recall', sum(rec for rec in recalls.values()) / len(recalls))
