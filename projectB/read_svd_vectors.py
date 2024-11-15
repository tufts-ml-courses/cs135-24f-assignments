
import pandas as pd
from surprise import Dataset, NormalPredictor, Reader, SVD, accuracy
from surprise.model_selection import cross_validate
import numpy as np

"""
Demo file showing how to extract vector representations with suprise models. 
"""

from train_valid_test_loader import load_train_valid_test_datasets

# Load the dataset in the same way as the main problem 
train_tuple, valid_tuple, test_tuple, n_users, n_items = \
        load_train_valid_test_datasets()


def tuple_to_surprise_dataset(tupl):
    """
    This function convert a subset in the tuple form to a `surprise` dataset. 
    """
    ratings_dict = {
        "userID": tupl[0],
        "itemID": tupl[1],
        "rating": tupl[2],
    }

    df = pd.DataFrame(ratings_dict)

    # A reader is still needed but only the rating_scale param is requiered.
    reader = Reader(rating_scale=(1, 5))

    # The columns must correspond to user id, item id and ratings (in that order).
    dataset = Dataset.load_from_df(df[["userID", "itemID", "rating"]], reader)

    return dataset

## Below we train an SVD model and get its vectors 

# train an SVD model using the training set
trainset = tuple_to_surprise_dataset(train_tuple).build_full_trainset()
algo = SVD()
algo.fit(trainset)


# Use an example to show to to slice out user and item vectors learned by the SVD 
uid = valid_tuple[0][0]
iid = valid_tuple[1][0]
rui = valid_tuple[2][0]

# Get model parameters
# NOTE: the SVD model has its own index system because the storage using raw user and item ids
# is not efficient. We need to convert raw ids to internal ids. Please read the few lines below
# carefully 

mu = algo.trainset.global_mean # SVD does not even fit mu -- it directly use the rating mean 
bu = algo.bu[trainset.to_inner_uid(uid)]
bi = algo.bi[trainset.to_inner_iid(iid)] 
pu = algo.pu[trainset.to_inner_uid(uid)] 
qi = algo.qi[trainset.to_inner_iid(iid)]

# Sanity check: we compute our own prediction and compare it against the model's prediction 
# our prediction
my_est = mu + bu + bi + np.dot(pu, qi) 

# the model's prediction
# NOTE: the training of the SVD model is random, so the prediction can be different with 
# different runs -- this is normal.   
svd_pred = algo.predict(uid, iid, r_ui=rui)

# The two predictions should be the same
print("My prediction: " + str(my_est) + ", SVD's prediction: " + str(svd_pred.est) + ", difference: " + str(np.abs(my_est - svd_pred.est)))

assert(np.abs(my_est - svd_pred.est) < 1e-6)


