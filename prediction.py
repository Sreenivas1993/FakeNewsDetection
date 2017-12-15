from kafka import KafkaConsumer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import operator
import numpy as np
import re
from bs4 import BeautifulSoup
from urllib import urlopen
from sklearn import model_selection
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from pyspark.mllib.util import MLUtils
from sklearn.externals import joblib
# $example off$
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from pyspark import SparkConf,SparkContext
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.mllib.linalg import Vectors
import sys
import rrealtime
import random

# Define SparkContext
conf = SparkConf().setAppName("Prediction").setMaster("local")
sc = SparkContext(conf = conf)


def g(x):
	print(x)
	print("Next")


formatted = pd.read_csv('datafiles/TrainData.csv')
# print(formatted.shape)
df = formatted.fillna(formatted.mean())

test = df[['Body', 'Title']]

test.to_csv("datafiles/Read.csv", index=False, encoding='utf-8')
print(test.head())

rdd = sc.textFile("datafiles/Read.csv")

rdd = rdd.map(lambda line: line.encode('utf-8'))
rdd.foreach(g)



# print(df.head())
# df = df.sample(20, axis =0)
# print(df.shape)





# X = df.drop("Label", axis =1)

# y = df["Label"]