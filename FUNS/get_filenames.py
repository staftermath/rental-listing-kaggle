import os
import sys
import pandas as pd
import numpy as np

#"/mnt/Movies-Freenas/images/"
if (len(sys.argv) > 1):
	rootFolder = sys.argv[1]
else:
	rootFolder = "/mnt/Movies-Freenas/images/"
if (len(sys.argv) > 2):
	maxLength = int(sys.argv[2])
else:
	maxLength = None
train_path = "/home/weiwen/Documents/projects/Kaggle/rental_listing_inquiries/data/train.json"
test_path = "/home/weiwen/Documents/projects/Kaggle/rental_listing_inquiries/data/test.json"
trainDT = pd.read_json(train_path)
testDT = pd.read_json(test_path)
listing_id = np.append(trainDT['listing_id'].ravel(),testDT['listing_id'].ravel())
print("Total Id Number: {}".format(len(listing_id)))

import pyspark

sc = pyspark.SparkContext("local[*]", "myApp")
def GetImageFileList(lst_id, rootFolder):
    folder = os.path.join(rootFolder, str(lst_id))
    if os.path.isdir(folder):
        return [os.path.join(folder, file) for file in os.listdir(folder)]
    else:
        return []

from functools import partial
GetImageFileFromTargetFolder = partial(GetImageFileList, rootFolder=rootFolder)

import time

if not maxLength:
	listing_id_sc = sc.parallelize(listing_id)
else:
	listing_id_sc = sc.parallelize(listing_id[:maxLength])
ts = time.time()
imagePathList = listing_id_sc.flatMap(lambda x:GetImageFileFromTargetFolder(x)).collect()
te = time.time()
print("Took {:.4f} secs".format(te-ts))

import pickle
cwd = os.getcwd()
with open(os.path.join(cwd, "fullImageFileName.pickle"), "wb") as file:
	pickle.dump(imagePathList, file)

print("Save File to {}".format(os.path.join(cwd, "fullImageFileName.pickle")))