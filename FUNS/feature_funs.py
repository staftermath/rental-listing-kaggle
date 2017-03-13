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
from datetime import datetime
import pdb

class GenerateDescriptionFeature():
    """
    Generate Col features
    """
    def __init__(self, fromCol, dictionary):
        self.dict = copy.copy(dictionary)
        self.fromCol = fromCol
    
    def transform(self, DF):
        """
        Add feature cols
        """
        print("Creating Feature from unit features")
        for string in self.dict.keys():
            PartialSearch = partial(self._SearchStringInRow, string=string)
            DF[self.dict[string]] = DF[[self.fromCol]].applymap(PartialSearch)
        return DF
    
    def fit(self, DF):
        return self.transform(DF)
    
    def _SearchStringInRow(self, entry, string):
        """
        Search if any element in an entry contains string pattern
        """
        return any(list(filter(lambda x:bool(re.search(string,x.lower())), entry)))


class GenerateNearestNeighborFeature():
    """
    Find the nearest neighbor and calculate the desired statistics within neighbors
    """
    def __init__(self, initDF, positionCol, valueCol, maxDist, funs=np.mean, metric='chebyshev'):
        self.position = copy.copy(initDF[positionCol])
        self.value = copy.copy(initDF[valueCol])
        self.positionCol = positionCol
        self.valueCol = valueCol
        self.metric = metric
        self.maxDist = maxDist
        self.funs = funs
        
    def transform(self, DF):
        """
        Calculate neighbors
        """
        print("Generate nearest neighbor features")
        pairwise = distance.cdist(np.array(DF[self.positionCol]), \
                          np.array(self.position), metric=self.metric)
        print("Pairwise distance calculated")
        indexNeighbor = [x < self.maxDist for x in pairwise]
        print("Neightbor list obtained")
        del pairwise
        for col in self.valueCol:
	        newColName = 'feature_{0}_{1}_nbr_{2}'.format(col, self.funs.__name__, self.metric)
	        print("New Column Added: {0}".format(newColName))
	        DF[newColName] = [self.funs(DF[x][col]) for x in indexNeighbor]
        return DF
    
    def fit(self, DF):
        return self.transform(DF)

class GenerateRawColAsFeature():
    """
    Simply Add New col with suitable names as feature
    """
    def __init__(self, fromcols, featureColHeader = "feature_"):
        self.featureColHeader = featureColHeader
        self.fromcols = fromcols
    
    def transform(self, DF):
        print("Use Raw Columns as Feature")
        newNameCols = [self.featureColHeader + x for x in self.fromcols]
        DF[newNameCols] = DF[self.fromcols]
        return DF
    
    def fit(self, DF):
        return self.transform(DF)

class KeepFeature():
    """
    Only keep feature Cols
    """
    def __init__(self, featureColHeader = "feature_"):
        self.featureColHeader = featureColHeader
        
    def transform(self, DF):
        keepCols = list(filter(lambda x:re.search("^{0}".format(self.featureColHeader), x), list(DF.columns)))
        return DF[keepCols]
    
    def fit(self, DF):
        return self.transform(DF)

class GenerateRowWiseFeature():
    """
    Only keep feature Cols
    """
    def __init__(self, fromcols, operations, identifier, featureColHeader = "feature_"):
        self.featureColHeader = featureColHeader
        self.fromcols = ["DF.{0}".format(x) for x in fromcols]
        slots = [m.group(0) for m in re.finditer(r"\{(\w*)\}", operations)]
        self.operations = operations
        assert len(set(slots)) ==  len(fromcols)
        self.identifier = identifier
        
    def transform(self, DF):
    	featureCol = eval(self.operations.format(*self.fromcols))
    	DF[self.featureColHeader+self.identifier] = featureCol
    	return DF
    
    def fit(self, DF):
        return self.transform(DF)

class GenerateMappingFeature():
	"""
	Generate Feature that is True if fromCol is equal to certain Value
	"""
	def __init__(self, fromcols, mapFuns, featureColHeader="feature_"):
		self.fromcols = fromcols
		self.mapFuns = mapFuns
		self.featureColHeader = featureColHeader

	def transform(self, DF):
		for col in self.fromcols:
			newFeatureName = self.featureColHeader + col + \
							'_' + self.mapFuns.__name__
			print("Adding New Col: "+newFeatureName)
			DF[newFeatureName] = list(map(self.mapFuns, DF[col]))
		return DF

	def fit(self, DF):
		return self.transform(DF)


class GenerateRollupFeature():
	"""
	Generate Roll Up values from train data
	"""
	def __init__(self, initDF, datetimeCol, fromCols, \
				 mapFuns, \
				 rollupWindow, \
				 fillna = 0,\
				 min_periods = 1, \
				 featureColHeader="feature_", \
				 datetimeFormat="%Y-%m-%d %H:%M:%S"):
		self.datetimeCol = datetimeCol
		self.fromCols = fromCols
		self.mapFuns = mapFuns
		self.rollupWindow = rollupWindow
		self.fillna = fillna
		self.min_periods = min_periods
		self.featureColHeader = featureColHeader
		self.datetimeFormat = datetimeFormat
		if not isinstance(initDF[self.datetimeCol][0], datetime):
			initDF[self.datetimeCol] = list(map(lambda x:datetime.strptime(x, self.datetimeFormat), initDF[self.datetimeCol]))
		storedDF = pd.DataFrame({self.fromCols:initDF[fromCols].values}, \
								index=pd.DatetimeIndex(initDF[self.datetimeCol]))
		self.initDF = storedDF.assign(stored=True)

	def transform(self, DF):
		print("Generating Feature "+self.showFeatureName())
		if not isinstance(DF[self.datetimeCol][0], datetime):
			DF[self.datetimeCol] = list(map(lambda x:datetime.strptime(x, self.datetimeFormat), DF[self.datetimeCol]))
		tempDF = pd.DataFrame({self.fromCols:DF[self.fromCols].values}, \
							  index=pd.DatetimeIndex(DF[self.datetimeCol]))
		concatenated = pd.concat([self.initDF, tempDF.assign(stored=False)])
		concatenated = concatenated.fillna(self.fillna).rolling(window=self.rollupWindow, min_periods=self.min_periods)
		concatenated = getattr(concatenated, self.mapFuns)()
		concatenated = concatenated[concatenated['stored'] == False]
		del concatenated['stored']
		concatenated.rename(columns={self.fromCols:self.showFeatureName()}, inplace=True)
		DF = pd.merge(DF, concatenated,  left_on=self.datetimeCol, right_index=True)
		return DF

	def fit(self, DF):
		return self.transform(DF)

	def showFeatureName(self):
		return self.featureColHeader+self.fromCols+'_rollup_'+str(self.rollupWindow)+'_days'



