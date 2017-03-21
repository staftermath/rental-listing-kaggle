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
import re
import string
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
        indexNeighbor = [np.logical_and(x < self.maxDist, x > 0) for x in pairwise]
        print("Neightbor list obtained")
        del pairwise
        for col in self.valueCol:
	        newColName = 'feature_{0}_{1}_nbr_{2}'.format(col, self.funs.__name__, self.metric)
	        print("New Column Added: {0}".format(newColName))
	        DF[newColName] = [self.funs(self.value[col][x]) for x in indexNeighbor]
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
		self.initDF = None
		self._initRollUp(initDF)

	def transform(self, DF):
		print("Generating Feature "+self.showFeatureName())
		if not isinstance(DF[self.datetimeCol][0], datetime):
			DF[self.datetimeCol] = list(map(lambda x:datetime.strptime(x, self.datetimeFormat), DF[self.datetimeCol]))
		indexCol = DF[self.datetimeCol].values
		storedDF = pd.DataFrame({self.fromCols:np.NaN, "stored":False, "time":indexCol}, index=DF.index)
		concatenated = pd.concat([self.initDF, storedDF])
		concatenated.sort_values(by="time", inplace=True)
		filled = concatenated.assign(**{self.fromCols:concatenated[self.fromCols].fillna(method="ffill")})
		filled = filled[filled['stored']==False]
		print(len(filled))
		filled.rename(columns={self.fromCols:self.showFeatureName()}, inplace=True)
		# DF = pd.merge(DF, concatenated,  left_on=self.datetimeCol, right_index=True)
		return DF.merge(filled[[self.showFeatureName()]], left_index=True, right_index=True)

	def _initRollUp(self, DF):
		indexCol = DF[self.datetimeCol].values
		if not isinstance(DF[self.datetimeCol][0], datetime):
			indexCol = list(map(lambda x:datetime.strptime(x, self.datetimeFormat), indexCol))
		storedDF = pd.DataFrame({self.fromCols:DF[self.fromCols].values, "time":pd.DatetimeIndex(indexCol)}, index=DF.index)
		storedDF.sort_values(by="time", inplace=True)
		storedDF_roll = storedDF.rolling(window=self.rollupWindow, min_periods=self.min_periods, on="time")
		self.initDF = getattr(storedDF_roll, self.mapFuns)()
		self.initDF.assign(stored=True)

	def fit(self, DF):
		return self.transform(DF)

	def showFeatureName(self):
		return self.featureColHeader+self.fromCols+'_rollup_'+str(self.rollupWindow)+'_days'

class GenerateGroupStats():
	"""
	calculate cols stats by group
	"""
	def __init__(self, initDF, fromCols, groupby, FUNS, actionOnRare, min_counts=10, featureColHeader="feature_"):
		self.fromCols = fromCols
		self.groupby = groupby
		self.FUNS = FUNS
		self.min_counts=min_counts
		self.actionOnRare=actionOnRare
		self.featureColHeader = featureColHeader
		self._commonGroup=None
		self._rareGroupValue=None

		self._initGroup(DF=initDF, groupby=self.groupby, fromCols=self.fromCols, min_counts=self.min_counts, actionOnRare=self.actionOnRare)

	def _initGroup(self, DF, groupby, fromCols, actionOnRare, min_counts=10):
		# Get Mgr list count
		groupDF = DF.groupby(groupby)[self.fromCols].agg(self.FUNS)
		countDF = DF.groupby(groupby)[self.fromCols].count().iloc[:, [0]]
		countDF.columns = ["counts"]
		groupDF = groupDF.merge(countDF, left_index=True, right_index=True)
		self._commonGroup = groupDF[groupDF["counts"] >= min_counts]
		if actionOnRare == np.NaN:
			self._rareGroupValue = pd.Series([np.NaN]*len(fromCols), index=fromCols)
		else:
			self._rareGroupValue = actionOnRare(DF.loc[~DF[groupby].isin(self._commonGroup.index), fromCols])

	def transform(self, DF):
		# Get new feature
		for col in self.fromCols:
			newColName = "{0}{4}_groupby_{1}_{2}_min_{3}".format(self.featureColHeader,self.groupby,self.FUNS.__name__,self.min_counts,col)
			print("Generating Feature {0}".format(newColName))
			newCol = DF[[self.groupby]].merge(self._commonGroup[[col]], how='left', left_on=self.groupby, right_index=True)
			newCol.rename({col:newColName}, inplace=True)
			newCol.fillna(self._rareGroupValue[col], inplace=True)
			DF = DF.assign(**{newColName:newCol[[col]]})
		return DF

	def fit(self, DF):
		return transform(DF)


class GenerateDateTimeFeature():
	"""
	Generate features from date time
	"""
	def __init__(self, fromCol, dataTimeFormat="%Y-%m-%d %H:%M:%S", extract=["month"], featureColHeader="feature_"):
		self.fromCol = fromCol
		self.featureColHeader = featureColHeader
		self.dataTimeFormat = dataTimeFormat
		self.extract = extract[:]
		self.extractFuns = [self.getAttr(attr) for attr in self.extract]

	def _strptime(self, x):
		return datetime.strptime(x, self.dataTimeFormat)

	def transform(self, DF):
		dateTime = DF[[self.fromCol]].applymap(self._strptime)
		for key, attr in enumerate(self.extract):
			newColName = "{0}time_{1}".format(self.featureColHeader, attr)
			print("Generating feature {0}".format(newColName))
			newCol = dateTime[[self.fromCol]].applymap(self.extractFuns[key])
			DF = DF.assign(**{newColName:newCol[[self.fromCol]]})
		return DF


	def getAttr(self, attr):
		def f(x):
			return getattr(x, attr)
		return f

	def fit(self, DF):
		return self.transform(DF)