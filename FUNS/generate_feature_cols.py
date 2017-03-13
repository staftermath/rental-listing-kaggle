from sklearn.pipeline import Pipeline
from functools import partial
import numpy as np
import re
import pandas as pd
import xgboost
import pickle
import copy
from scipy.spatial import distance
from functools import partial
import re,string
import json
from feature_funs import *

train_path = "/home/weiwen/Documents/projects/Kaggle/rental_listing_inquiries/data/train.json"
with open(train_path) as file:
    train = json.load(file)

def ConvertJsonToDF(json, cols=None):
    if cols:
        assert cols < list(json.keys())
    else:
        cols = list(json.keys())
    # Validation Fun
    rowKey = list(json[cols[0]].keys())
    returnDF = pd.DataFrame(index=rowKey, columns=cols)
    for col in cols:
        returnDF[col] = list(json[col].values())
    return returnDF

trainDF = ConvertJsonToDF(train)
interest_cat_to_int = {'medium':1, 'low':0, 'high':2}
trainDF['interest_level'] = [interest_cat_to_int[x] for x in trainDF['interest_level']]

featureDict = {"laundry":"feature_laundry", \
               "prewar|pre-war":"feature_prewar", \
               "^cats* | cats* |^dogs* | dogs* |^pets*| pets* !no":"feature_petfriendly", \
               "no fee":"feature_nofee", \
               "elevator":"feature_elevator", \
               "wood":"feature_woodenfloor", \
               "garden|patio":"feature_gardenpatio", \
               "dishwasher":"feature_dishwasher", \
               "fitness":"feature_fitness", \
               "dining":"feature_diningroom", \
               "pool":"feature_pool", \
               "garage":"feature_garage", \
               "doorman":"feature_doorman"}
GenerateDescriptionFeatureParamed = GenerateDescriptionFeature(dictionary=featureDict, fromCol='features')

GenerateNearestNeighborFeatureParamed = GenerateNearestNeighborFeature(initDF=trainDF, \
                                                                       positionCol=['latitude', 'longitude'], \
                                                                       valueCol=['interest_level','price'], \
                                                                       maxDist=0.001
                                                                       )

GenerateRawColAsFeatureParamed = GenerateRawColAsFeature(fromcols=["bathrooms", "bedrooms", "price"])
GenerateRowWiseFeatureParamed = GenerateRowWiseFeature(fromcols=['feature_price','feature_price_mean_nbr_chebyshev'], \
							 operations='{0}-{1}', identifier='price_diff')
GenerateMappingFeatureParamed = GenerateMappingFeature(fromcols=['photos'], \
							mapFuns=len)
GenerateRollupFeatureParamed = GenerateRollupFeature(initDF=trainDF, \
                           datetimeCol='created', \
                           fromCols='price', \
                           mapFuns='mean', \
                           rollupWindow=7, \
                           featureColHeader="intermediate_")
GenerateHistoricalPriceDiffParamed = GenerateRawColAsFeature(fromcols=["intermediate_price_rollup_7_days", "price"])
KeepFeatureParamed = KeepFeature()

modelPipeline = Pipeline([('unitfeature', GenerateDescriptionFeatureParamed), \
         ('nearestneighbor', GenerateNearestNeighborFeatureParamed), \
         ('rawcol', GenerateRawColAsFeatureParamed), \
         ('priceCompare', GenerateRowWiseFeatureParamed), \
         ('numPhotos', GenerateMappingFeatureParamed), \
         ('historicalPrice', GenerateRollupFeatureParamed), \
         ('historicalPriceDiff', GenerateHistoricalPriceDiffParamed), \
         ('filtercol', KeepFeatureParamed)])


train_X = modelPipeline.transform(trainDF)
train_y = trainDF['interest_level']

with open("train_X.pickle", "wb") as file:
	pickle.dump(train_X, file)

print("Feature Generation Complete")

import xgboost

xgb = xgboost.XGBClassifier(max_depth=3, n_estimators=200, learning_rate=0.05)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_X, train_y)

print("Training Model...")

xgb.fit(X=X_train, y=y_train)

print("Saving Model Object...")

with open("model.pickle", 'wb') as file:
	pickle.dump(xgb, file)

varImp = pd.DataFrame()
varImp['Variables'] = list(X_train.columns)
varImp['Importance'] = xgb.feature_importances_
varImp = varImp.sort_values(by='Importance', ascending=False)

print("Saving Variable Importance")
varImp.to_csv("Variable_Importance.csv", index=False)

print("Making Prediction")
y_test_pred = xgb.predict(X_test)

# calculate log loss
y_test_prob = xgb.predict_proba(X_test)
y_test_array = np.array(pd.get_dummies(y_test))
logloss= -np.sum(np.multiply(y_test_array, np.log(y_test_prob)))/len(y_test)
print("Mean Log Loss: {}".format(logloss))

failedPrediction = trainDF.loc[y_test.index[y_test_pred!=y_test]]
failedPrediction['predicted'] = y_test_pred[y_test_pred!=y_test]
failedPrediction.to_csv('failed_prediction.csv', index=False)