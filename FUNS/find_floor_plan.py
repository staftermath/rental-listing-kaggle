import pyspark
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
from PIL import ImageFile
import os
import sys
import pickle
import time

if len(sys.argv) > 1:
	maxLength = int(sys.argv[1])
else:
	maxLength = None

ImageFile.LOAD_TRUNCATED_IMAGES = True
img_dir="/mnt/Movies-Freenas/images/"

train_path = "/home/weiwen/Documents/projects/Kaggle/rental_listing_inquiries/data/train.json"
test_path = "/home/weiwen/Documents/projects/Kaggle/rental_listing_inquiries/data/test.json"
trainDT = pd.read_json(train_path)
testDT = pd.read_json(test_path)
listing_id = np.append(trainDT['listing_id'].ravel(),testDT['listing_id'].ravel())

del trainDT
del testDT

def VarFromStackedArray(arr, count):
    mean = arr*count[:, np.newaxis]/np.sum(count)
    return np.sum((arr - mean)**2*count[:, np.newaxis])/np.sum(count)

def Entropy(data):
    total = np.sum(list(data.values()))
    valueArray = np.array(list(data.values()))/total
    return -np.sum(valueArray*np.log(valueArray))

def AnalysiseListingId(lst_id, img_dir=None):
    currentdir = os.path.join(img_dir, str(lst_id))
    if not os.path.isdir(currentdir):
        return([lst_id, None, None, None, None,None, None])
    returnList=[]
    try:
        for imgName in os.listdir(currentdir):
            img_url = os.path.join(currentdir, imgName)
            im = Image.open(img_url)  
            w, h = im.size  
            colors = im.getcolors(w*h)
            variance = VarFromStackedArray(np.array([list(x[1]) for x in colors]), \
                                           np.array([x[0] for x in colors]))
            colorGrid = {}
            for pixel in colors:
                colorKey = "{0}_{1}_{2}".format(*(x//20 for x in pixel[1]))
                colorGrid[colorKey] = colorGrid.get(colorKey, 0) + pixel[0]
            entropy = Entropy(colorGrid)    
            returnList.append([lst_id, imgName, w, h, len(colors), variance, entropy])
        return(returnList)
    except:
        return([lst_id, None, None, None, None, None, None])

from functools import partial
analysisToSpark = partial(AnalysiseListingId, img_dir=img_dir)

sc = pyspark.SparkContext("local[*]", "myApp")
if not maxLength:
	listing_id_spark = sc.parallelize(listing_id)
else:
	listing_id_spark = sc.parallelize(listing_id[:maxLength])
ts = time.time()
getAnalysis=listing_id_spark.flatMap(lambda x: analysisToSpark(lst_id=x)).collect()
te = time.time()
print("Took {} sec".format(te-ts))
print(getAnalysis[:5])
with open("photoAnalysis.pickle", "wb") as file:
	pickle.dump(getAnalysis, file)