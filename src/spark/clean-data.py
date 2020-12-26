from pyspark.sql import SparkSession
spark = SparkSession.builder.master("spark://192.168.0.111:7077").appName('Zootopia-clean-data').getOrCreate()
from pyspark import SparkContext, SparkConf
sc = SparkContext.getOrCreate()	

PATH_24H = 'hdfs://192.168.0.111:9000/data/news_24h.csv'
PATH_LAWS = 'hdfs://192.168.0.111:9000/data/news_laws.csv'
PATH_SOHA = 'hdfs://192.168.0.111:9000/data/news_soha.csv'

df_24h_spark = spark.read.load(PATH_24H,format="csv", sep=",", inferSchema="true", header="true")
df_laws_spark = spark.read.load(PATH_LAWS,format="csv", sep=",", inferSchema="true", header="true")
df_soha_spark = spark.read.load(PATH_SOHA,format="csv", sep=",", inferSchema="true", header="true")

from pyspark.sql.functions import col
df_24h_spark = df_24h_spark.filter(col("content").isNotNull()).filter(col("topic").isNotNull())
df_laws_spark = df_laws_spark.filter(col("content").isNotNull()).filter(col("topic").isNotNull())
df_soha_spark = df_soha_spark.filter(col("content").isNotNull()).filter(col("topic").isNotNull())


df_laws_spark = df_laws_spark.withColumnRenamed("description", "sapo")
df_soha_spark = df_soha_spark.withColumnRenamed("description", "sapo")

from pyspark.sql.functions import lit
df_24h_spark = df_24h_spark.withColumn("sub_topic", lit("missing"))
df_soha_spark = df_soha_spark.withColumn("sub_topic", lit("missing"))

df_all_news = df_24h_spark.union(df_laws_spark).union(df_24h_spark)
