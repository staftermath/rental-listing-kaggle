from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, log_loss
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
test_path = "/home/weiwen/Documents/projects/Kaggle/rental_listing_inquiries/data/test.json"
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

from sklearn.model_selection import train_test_split

train_y = trainDF['interest_level']

X_train, X_test, y_train, y_test = train_test_split(trainDF, train_y, test_size=0.2, random_state=123)
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)

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

GenerateNearestNeighborFeatureParamed = GenerateNearestNeighborFeature(initDF=X_train, \
                                                                       positionCol=['latitude', 'longitude'], \
                                                                       valueCol=['interest_level','price'], \
                                                                       maxDist=0.001
                                                                       )

GenerateRawColAsFeatureParamed = GenerateRawColAsFeature(fromcols=["bathrooms", "bedrooms", "price"])
GenerateRowWiseFeatureParamed = GenerateRowWiseFeature(fromcols=['feature_price','feature_price_mean_nbr_chebyshev'], \
							 operations='{0}-{1}', identifier='price_diff')
FeatureLens = GenerateMappingFeature(fromcols=['features'], \
              mapFuns=len)
GenerateMappingFeatureParamed = GenerateMappingFeature(fromcols=['photos'], \
							mapFuns=len)
GenerateGroupMgrStatsParamed = GenerateGroupStats(initDF=X_train, fromCols=["price","interest_level"], \
                                                         groupby="manager_id", FUNS=np.mean, \
                                                         actionOnRare=np.mean)
GenerateGroupBuildStatsParamed = GenerateGroupStats(initDF=X_train, fromCols=["price","interest_level"], \
                                                         groupby="building_id", FUNS=np.mean, \
                                                         actionOnRare=np.mean)
GenerateMgrPriceDiff = GenerateRowWiseFeature(fromcols=['feature_price','feature_price_groupby_manager_id_mean_min_10'], \
               operations='{0}-{1}', identifier='mgr_price_diff')
GenerateBuildPriceDiff = GenerateRowWiseFeature(fromcols=['feature_price','feature_price_groupby_building_id_mean_min_10'], \
               operations='{0}-{1}', identifier='building_price_diff')
GeneratePriceBdrRatio = GenerateRowWiseFeature(fromcols=['price','bedrooms'], \
               operations='{0}/{1}', identifier='price_bedroom_ratio')
GeneratePriceBthRatio = GenerateRowWiseFeature(fromcols=['price','bathrooms'], \
               operations='{0}/{1}', identifier='price_bathrooms_ratio')

GenerateMthDay = GenerateDateTimeFeature(fromCol="created", extract=['month', 'day'])
# GenerateRollupFeatureParamed = GenerateRollupFeature(initDF=trainDF, \
#                            datetimeCol='created', \
#                            fromCols='price', \
#                            mapFuns='mean', \
#                            rollupWindow=7, \
#                            featureColHeader="intermediate_")
# GenerateHistoricalPriceDiffParamed = GenerateRowWiseFeature(fromcols=["intermediate_price_rollup_7_days", "price"], \
#                             operations='{0}-{1}', identifier='history_price_diff')
KeepFeatureParamed = KeepFeature()

modelPipeline = Pipeline([('unitfeature', GenerateDescriptionFeatureParamed), \
         ('nearestneighbor', GenerateNearestNeighborFeatureParamed), \
         ('rawcol', GenerateRawColAsFeatureParamed), \
         ('priceCompare', GenerateRowWiseFeatureParamed), \
         ('numPhotos', GenerateMappingFeatureParamed), \
         ('manager', GenerateGroupMgrStatsParamed), \
         ('building', GenerateGroupBuildStatsParamed), \
         ('mgrPriceDiff', GenerateMgrPriceDiff), \
         ('buildPriceDiff', GenerateBuildPriceDiff), \
         ('mthday', GenerateMthDay), \
         ('prbrdRatio', GeneratePriceBdrRatio), \
         ('prbthRatio', GeneratePriceBthRatio), \
         # ('historicalPrice', GenerateRollupFeatureParamed), \
         # ('historicalPriceDiff', GenerateHistoricalPriceDiffParamed), \
         ('filtercol', KeepFeatureParamed)])

from sklearn.model_selection import cross_val_score
X_train_transformed = modelPipeline.transform(X_train)
X_test_transformed = modelPipeline.transform(X_test)
# X_val_transformed = modelPipeline.transform(X_val)
with open("train_X.pickle", "wb") as file:
	pickle.dump(X_train_transformed, file)

print("Feature Generation Complete")

import xgboost

# def logloss(y_true, y_pred):
#   y_test_array = np.array(pd.get_dummies(y_true))
#   logloss= -np.sum(np.multiply(y_test_array, np.log(y_pred)))/len(y_true)
#   return logloss

logloss_scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
# xgb = xgboost.XGBClassifier(max_depth=5, n_estimators=200, learning_rate=0.1)

# param_grid = {"max_depth":[5, 7], \
#               "n_estimators":[300, 500], \
#               "learning_rate":[0.05, 0.1, 0.2]}
# xgb = xgboost.XGBClassifier()
# xgb_CV = GridSearchCV(xgb, param_grid = param_grid, scoring=logloss_scorer, n_jobs=-1, cv=3, verbose=1)

print("Training Model...")

# xgb_CV.fit(X=X_train_transformed, y=y_train)
xgb = xgboost.XGBClassifier(max_depth=5, n_estimators=500, learning_rate=0.05, colsample_bytree=0.8)
# xgb = xgboost.XGBRegressor(max_depth=5, n_estimators=500, learning_rate=0.05, colsample_bytree=0.8)
xgb.fit(X=X_train_transformed, y=y_train)

print("Saving Model Object...")

with open("model.pickle", 'wb') as file:
  pickle.dump(xgb, file)


varImp = pd.DataFrame()
varImp['Variables'] = list(X_train_transformed.columns)
# varImp['Importance'] = xgb_CV.best_estimator_.feature_importances_
varImp['Importance'] = xgb.feature_importances_
varImp = varImp.sort_values(by='Importance', ascending=False)

print("Saving Variable Importance")
varImp.to_csv("Variable_Importance.csv", index=False)
# print("Refit using Full data")
# xgb_CV.best_estimator_.fit(X=X_train_transformed, y=y_train)
print("Making Prediction")
# y_test_pred = xgb_CV.best_estimator_.predict(X_test_transformed)
y_test_pred = xgb.predict(X_test_transformed)

# calculate log loss
y_test_prob = xgb.predict_proba(X_test_transformed)
y_test_array = np.array(pd.get_dummies(y_test))
logloss= -np.sum(np.multiply(y_test_array, np.log(y_test_prob)))/len(y_test)
print("Mean Log Loss: {}".format(logloss))

failedPrediction = X_test.loc[y_test.index[y_test_pred!=y_test]]
failedPrediction['predicted'] = y_test_pred[y_test_pred!=y_test]
failedPrediction.to_csv('failed_prediction.csv', index=False)

with open(test_path) as file:
    test = json.load(file)

testDF = ConvertJsonToDF(test)

X_test_final = modelPipeline.transform(testDF)
y_test_final = xgb.predict_proba(X_test_final)
y_test_finalDT = pd.DataFrame(y_test_final)
y_test_finalDT.columns = ['low','medium','high']
y_test_finalDT = y_test_finalDT.assign(listing_id=testDF["listing_id"].values)
with open('y_test_final.pickle', 'wb') as file:
  pickle.dump(y_test_finalDT, file)


y_test_finalDT.to_csv('submission.csv', index=False)
