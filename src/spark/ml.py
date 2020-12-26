#!/usr/bin/python
# -*- coding: utf-8 -*-
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("spark://192.168.0.111:7077").appName('Zootopia-ML').getOrCreate()
from pyspark import SparkContext
sc = SparkContext.getOrCreate()
import re
from pyspark.sql.types import StringType, ArrayType, StructField
from pyvi import ViTokenizer
def normalize(text):
	ViTokenizer.tokenize(text)
	pattern_keep = r"[\w_ ]+"
	keep_char = " ".join(re.findall(pattern_keep, text))
	keep_char = re.sub("\d+", " ", keep_char)
	text_normalized = re.sub("\s+", " ", keep_char).strip().lower()
	words_result = text_normalized.split()
	return words_result

# make normalize become udf - user define function
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, ArrayType

PATH_24H = 'hdfs://192.168.0.111:9000/data/news_24h.csv'
df_24h_spark = spark.read.load(PATH_24H,format="csv", sep=",", inferSchema="true", header="true")
df_24h_spark = df_24h_spark.filter(col("content").isNotNull()).filter(col("topic").isNotNull())
normalizer = udf(lambda z: normalize(z), ArrayType(StringType()))
topic_use = df_24h_spark.groupBy("topic").count() \
    .orderBy(col("count").desc()) \
    .limit(20).select("topic")\
    .toPandas().values.tolist()
topic_use = [t[0] for t in topic_use]
df_24h_spark_clf = df_24h_spark.filter(col("topic").isin(topic_use))

# normalize:
df_24h_spark_clf = df_24h_spark_clf.withColumn("content_normalized", normalizer("content"))

from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import LogisticRegression, FMClassifier
# regular expression tokenizer
# regexTokenizer = RegexTokenizer(inputCol="content_normalized", outputCol="words", pattern="\\W")
# stop words
add_stopwords = ["http","https", u"à", u"và",u"ừ",u"nhỉ",u"nhé"]
stopwordsRemover = StopWordsRemover(inputCol="content_normalized", outputCol="content_normalized_final").setStopWords(add_stopwords)
# bag of words count
countVectors = CountVectorizer(inputCol="content_normalized_final", outputCol="features", vocabSize=10000, minDF=5)

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.feature import HashingTF, IDF
# Try tfidf is ok
label_stringIdx = StringIndexer(inputCol = "topic", outputCol = "label")
pipeline = Pipeline(stages=[stopwordsRemover, countVectors, label_stringIdx])
# Fit the pipeline to training documents.
pipelineFit = pipeline.fit(df_24h_spark_clf)
dataset = pipelineFit.transform(df_24h_spark_clf)

train_set, test_set = dataset.randomSplit([0.7, 0.3], seed = 99)

from pyspark.ml.classification import NaiveBayes, FMClassifier, RandomForestClassifier, OneVsRest, LinearSVC
model = NaiveBayes(smoothing=1.0, modelType="multinomial")

model1  = model.fit(train_set)
predictions = model1.transform(test_set)
# evaluation
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

#
print("accuracy: ", accuracy)


