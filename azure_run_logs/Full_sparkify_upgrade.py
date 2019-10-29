# Databricks notebook source
# Starter code
from pyspark.sql import SparkSession

import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import time

from pyspark.sql import Window
from pyspark.sql.functions import avg, col, concat, desc, explode, lit, min, max, split, udf, count, round, array, isnull, struct, rand
from pyspark.sql.types import ArrayType, StringType, IntegerType, LongType, FloatType, BooleanType

from pyspark import keyword_only
from pyspark.ml import Transformer, Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, DecisionTreeClassifier, LogisticRegression, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.linalg import Vectors


# COMMAND ----------

# Create spark session
spark = SparkSession \
    .builder \
    .appName("Sparkify application") \
    .getOrCreate()

# COMMAND ----------

def dataset_train_test_split(dataset, split_ratio, randomstate):
  '''
  
  Spliting dataset using the userId field. separating the UserID into the split_ratio ratio
  
  inputs:
  dataset     : dataset to be splitted according to the 'userId' field
  split_ratio : array, representing the ratio of userId assigned to train data and test data
  randomstate : randomstate for repeatability
  
  outputs:
  df_train    : fraction of the input dataset, containing userId assigned to train
  df_test     : fraction of the input dataset, containing userId assigned to test
  
  '''
  df_userId_train, _ = dataset.select('userId').dropDuplicates().randomSplit(split_ratio, randomstate)
  list_userId_train = [row.userId for row in df_userId_train.collect()]

  check_in_list = udf(lambda x: True if x in list_userId_train else False)

  df_train = dataset.filter(check_in_list(dataset.userId) == True)
  df_test = dataset.filter(check_in_list(dataset.userId) != True)
  
  #print('Split dataset completed')
  
  return df_train, df_test
  

# COMMAND ----------

def rebalance_dataset(dataset, label_name, randomstate):
  '''
  
  Spliting dataset using the userId field. separating the UserID into the split_ratio ratio
  
  inputs:
  dataset        : dataset to be rebalanced against the field 'label_name'
  label_name     : string, that point to the column name taht needs to be rebalance for 1's to be as numerous as 0's
  randomstate    : randomstate for repeatability
  
  outputs:
  return_dataset : new dataset equivalent in structure as dataset, but with equivalent number of 0's and 1's for label_name

  '''
  
  nb_x = dataset.groupBy(col(label_name)).count().orderBy(label_name).collect()
  ratio = nb_x[1][1] / nb_x[0][1]
  #print("Previous numbers of 0's = {}, number's of 1's = {}, and ratio = {:.3f}".format(nb_x[0][1], nb_x[1][1], ratio))
  
  #return_dataset = dataset.filter(col(label_name) == 1).union(dataset.filter(col(label_name) == 0).randomSplit([ratio, 1-ratio], randomstate)[0])
  
  #nb_x = return_dataset.groupBy(col(label_name)).count().orderBy(label_name).collect()
  #print("  Actual numbers of 0's = {}, number's of 1's = {}, and ratio = {:.3f}".format(nb_x[0][1], nb_x[1][1], nb_x[1][1] / nb_x[0][1]))
    
  #return return_dataset
  return dataset.filter(col(label_name) == 1).union(dataset.filter(col(label_name) == 0).randomSplit([ratio, 1-ratio], randomstate)[0])

# COMMAND ----------

def return_evaluation(model, df, label_name):
  # Make predictions.
  panda = model.transform(df).toPandas()

  tp = panda[(panda[label_name] == 1) & (panda.prediction == 1.0)].shape[0]
  tn = panda[(panda[label_name] == 0) & (panda.prediction == 0.0)].shape[0]
  fp = panda[(panda[label_name] == 0) & (panda.prediction == 1.0)].shape[0]
  fn = panda[(panda[label_name] == 1) & (panda.prediction == 0.0)].shape[0]

  try:
    r = float(tp) / (tp + fn)
  except:
    r='Unapplicatable'

  try:
    p = float(tp) / (tp + fp)
  except:
    p='Unapplicatable'

  try:
    a = float(tp + tn) / (tp+tn+fp+fn)
  except:
    a='Unapplicatable'

  try:
    f = 2*r*p / (p+r)
  except:
    f='Unapplicatable'
    
  return tp, tn, fp, fn, r, p, a, f  

# COMMAND ----------

def print_evaluation(model, df, label_name):
  '''
  Printing evaluation of a dataset against a training model

  inputs:
  model       : Pipelined model that as been trained. We will transform the dataset useing this model
  dataset     : dataset to be tested, we will predict the outcome of this dataset and kpi it
  label_name  : name of the column that contains the expected result of prediction

  outputs:
  None
  But we will print 9 Kpis
      Number of True positives
      Number of True Negatives
      Number of False Positives
      Number of False Negatives
      Total number of records
      Recall
      Precision
      Accuracy
      F1 score

  '''

  # Make predictions.
  panda = model.transform(df).toPandas()

  tp = panda[(panda[label_name] == 1) & (panda.prediction == 1.0)].shape[0]
  tn = panda[(panda[label_name] == 0) & (panda.prediction == 0.0)].shape[0]
  fp = panda[(panda[label_name] == 0) & (panda.prediction == 1.0)].shape[0]
  fn = panda[(panda[label_name] == 1) & (panda.prediction == 0.0)].shape[0]

  print('--------------------------------------------')
  print('             ', label_name)
  print('--------------------------------------------')
  print("True Positives  : {}".format(tp))
  print("True Negatives  : {}".format(tn))
  print("False Positives : {}".format(fp))
  print("False Negatives : {}".format(fn))
  print("Total           : {}".format(tp+tn+fp+fn))

  try:
    r = float(tp) / (tp + fn)
    print("Recall          : {:.2f} %".format(100.0 * r))
  except:
    print("Recall          : Unapplicatable")

  try:
    p = float(tp) / (tp + fp)
    print("Precision       : {:.2f} %".format(100.0 * p))
  except:
    print("Precision       : Unapplicatable")

  try:
    a = float(tp + tn) / (tp+tn+fp+fn)
    print("Accuracy        : {:.2f} %".format(100.0 * a))
  except:
    print("Accuracy        : Unapplicatable")

  try:
    f = 2*r*p / (p+r)
    print("F Score         : {:.2f}".format(1.0 * f))
  except:
    print("F Score         : Unapplicatable")
  print('--------------------------------------------')


# COMMAND ----------

class Data_bucketer(Transformer, HasInputCol, HasOutputCol):

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, bucket_size=None, number_of_bucket_for_training=None,
                 prediction_date=None, number_of_days_predicting_for=None, number_of_duplication=None,
                 date_interval=None):
        super(Data_bucketer, self).__init__()
        self.bucket_size = Param(self, "bucket_size", "")
        self.number_of_bucket_for_training = Param(self, "number_of_bucket_for_training", "")
        self.prediction_date = Param(self, "prediction_date", "")
        self.number_of_days_predicting_for = Param(self, "number_of_days_predicting_for", "")
        self.number_of_duplication = Param(self, "number_of_duplication", "")
        self.date_interval = Param(self, "date_interval", "")
        self._setDefault(bucket_size=set(), number_of_bucket_for_training=set(), prediction_date=set(),
                         number_of_days_predicting_for=set(), number_of_duplication=set(), date_interval=set())
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, bucket_size=None, number_of_bucket_for_training=None,
                  prediction_date=None, number_of_days_predicting_for=None, number_of_duplication=None,
                  date_interval=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setbucket_size(self, value):
        self._paramMap[self.bucket_size] = value
        return self

    def getbucket_size(self):
        return self.getOrDefault(self.bucket_size)

    def setnumber_of_bucket_for_training(self, value):
        self._paramMap[self.number_of_bucket_for_training] = value
        return self

    def getnumber_of_bucket_for_training(self):
        return self.getOrDefault(self.number_of_bucket_for_training)

    def setprediction_date(self, value):
        self._paramMap[self.prediction_date] = value
        return self

    def getprediction_date(self):
        return self.getOrDefault(self.prediction_date)

    def setnumber_of_days_predicting_for(self, value):
        self._paramMap[self.number_of_days_predicting_for] = value
        return self

    def getnumber_of_days_predicting_for(self):
        return self.getOrDefault(self.number_of_days_predicting_for)

    def setnumber_of_duplication(self, value):
        self._paramMap[self.number_of_duplication] = value
        return self

    def getnumber_of_duplication(self):
        return self.getOrDefault(self.number_of_duplication)

    def setdate_interval(self, value):
        self._paramMap[self.date_interval] = value
        return self

    def getdate_interval(self):
        return self.getOrDefault(self.date_interval)

    def _transform(self, dataset):
        bucket_size = self.getbucket_size()
        number_of_bucket_for_training = self.getnumber_of_bucket_for_training()
        prediction_date = self.getprediction_date()
        number_of_days_predicting_for = self.getnumber_of_days_predicting_for()
        number_of_duplication = self.getnumber_of_duplication()
        date_interval = self.getdate_interval()
        
        start_time = time.time()

        pass_not_null = udf(lambda x: x[0] if x[0] is not None else x[1], StringType())

        for iteration in range(number_of_duplication + 1):
            date_of_prediction = prediction_date - 86400000 * date_interval * iteration

            # generate the row table of features

           
            start_bucket = date_of_prediction - 86400000 * bucket_size
            end_bucket = date_of_prediction

            check_NextSong = udf(lambda x: 1 if x == 'NextSong' else 0, IntegerType())
            check_Home = udf(lambda x: 1 if x == 'Home' else 0, IntegerType())
            check_Thumbs_Up = udf(lambda x: 1 if x == 'Thumbs Up' else 0, IntegerType())
            check_Add_to_Playlist = udf(lambda x: 1 if x == 'Add to Playlist' else 0, IntegerType())
            check_Roll_Advert = udf(lambda x: 1 if x == 'Roll Advert' else 0, IntegerType())
            check_Add_Friend = udf(lambda x: 1 if x == 'Add Friend' else 0, IntegerType())
            check_Login = udf(lambda x: 1 if x == 'Login' else 0, IntegerType())
            check_Thumbs_Down = udf(lambda x: 1 if x == 'Thumbs Down' else 0, IntegerType())
            check_Downgrade = udf(lambda x: 1 if x == 'Downgrade' else 0, IntegerType())
            check_Help = udf(lambda x: 1 if x == 'Help' else 0, IntegerType())
            check_Settings = udf(lambda x: 1 if x == 'Settings' else 0, IntegerType())
            check_About = udf(lambda x: 1 if x == 'About' else 0, IntegerType())
            check_Upgrade = udf(lambda x: 1 if x == 'Upgrade' else 0, IntegerType())
            check_Save_Settings = udf(lambda x: 1 if x == 'Save Settings' else 0, IntegerType())
            check_Error = udf(lambda x: 1 if x == 'Error' else 0, IntegerType())
            check_Submit_Upgrade = udf(lambda x: 1 if x == 'Submit Upgrade' else 0, IntegerType())
            check_Submit_Downgrade = udf(lambda x: 1 if x == 'Submit Downgrade' else 0, IntegerType())
            check_Cancel = udf(lambda x: 1 if x == 'Cancel' else 0, IntegerType())
            check_Cancellation_Confirmation = udf(lambda x: 1 if x == 'Cancellation Confirmation' else 0, IntegerType())
            check_Register = udf(lambda x: 1 if x == 'Register' else 0, IntegerType())
            check_Submit_Registration = udf(lambda x: 1 if x == 'Submit Registration' else 0, IntegerType())

            time_between = udf(lambda x: (date_of_prediction - float(x))/86400000.0 if x is not None else 0.0, FloatType())
            check_length = udf(lambda x: x if x is not None else 0.0, FloatType())
            convert_level = udf(lambda x: 1 if x == 'paid' else 0, IntegerType())
            convert_gender = udf(lambda x: 1 if x == 'F' else 0, IntegerType())

            yes_or_no = udf(lambda x: 0 if x == 0 else 1, IntegerType())

            wU = Window.partitionBy('userId')
            wUS = Window.partitionBy(['userId', 'sessionId'])


            data = dataset.filter(dataset.ts < lit(end_bucket))\
                            .filter(dataset.ts >= lit(start_bucket))\
                            .withColumn('key1', time_between(dataset.registration))\
                            .withColumn('NS', check_NextSong(dataset.page))\
                            .withColumn('HO', check_Home(dataset.page))\
                            .withColumn('TU', check_Thumbs_Up(dataset.page))\
                            .withColumn('AP', check_Add_to_Playlist(dataset.page))\
                            .withColumn('RA', check_Roll_Advert(dataset.page))\
                            .withColumn('AF', check_Add_Friend(dataset.page))\
                            .withColumn('LO', check_Login(dataset.page))\
                            .withColumn('TD', check_Thumbs_Down(dataset.page))\
                            .withColumn('DO', check_Downgrade(dataset.page))\
                            .withColumn('HE', check_Help(dataset.page))\
                            .withColumn('SE', check_Settings(dataset.page))\
                            .withColumn('AB', check_About(dataset.page))\
                            .withColumn('UP', check_Upgrade(dataset.page))\
                            .withColumn('SS', check_Save_Settings(dataset.page))\
                            .withColumn('ER', check_Error(dataset.page))\
                            .withColumn('SU', check_Submit_Upgrade(dataset.page))\
                            .withColumn('SD', check_Submit_Downgrade(dataset.page))\
                            .withColumn('CA', check_Cancel(dataset.page))\
                            .withColumn('CC', check_Cancellation_Confirmation(dataset.page))\
                            .withColumn('RE', check_Register(dataset.page))\
                            .withColumn('SR', check_Submit_Registration(dataset.page))\
                            .withColumn('key5', check_length(dataset.length))\
                            .groupBy('userId').agg({'key1': 'max', 'NS': 'sum', 'HO': 'sum', 'TU': 'sum', 'AP': 'sum', 'RA': 'sum', 'AF': 'sum', 'LO': 'sum', 'TD': 'sum', 'DO': 'sum', 'HE': 'sum', 'SE': 'sum', 'AB': 'sum', 'UP': 'sum', 'SS': 'sum', 'ER': 'sum', 'SU': 'sum', 'SD': 'sum', 'CA': 'sum', 'CC': 'sum', 'RE': 'sum', 'SR': 'sum', 'key5': 'sum'})

            data1 = dataset.filter(dataset.ts < lit(end_bucket))\
                            .filter(dataset.ts >= lit(start_bucket))\
                            .withColumn('Max_itemInSession', max('itemInSession').over(wUS)).where(col('itemInSession') == col('Max_itemInSession'))\
                            .withColumnRenamed('itemInSession', 'key4')\
                            .groupBy('userId').agg({'key4': 'avg'})\
                            .withColumnRenamed('userId', 'userId_b')

            data2 = dataset.filter(dataset.ts < lit(end_bucket))\
                            .filter(dataset.ts >= lit(start_bucket))\
                            .withColumn('Max_ts', max('ts').over(wU)).where(col('ts') == col('Max_ts'))\
                            .withColumn('key2', convert_level(dataset.level))\
                            .withColumn('key6', convert_gender(dataset.gender))\
                            .select([col('userId').alias('userId_c')] + ['key2', 'key6'])


            start_bucket = date_of_prediction
            end_bucket = start_bucket + 86400000 * number_of_days_predicting_for
            label_columns = dataset.filter(dataset.ts < lit(end_bucket))\
                            .filter(dataset.ts >= lit(start_bucket))\
                            .withColumn('SU', check_Submit_Upgrade(dataset.page))\
                            .withColumn('SD', check_Submit_Downgrade(dataset.page))\
                            .withColumn('CC', check_Cancellation_Confirmation(dataset.page))\
                            .groupBy('userId')\
                            .max()\
                            .select(col('userId').alias('userId_d'), col('max(SU)').alias('upgrade'), col('max(SD)').alias('downgrade'), col('max(CC)').alias('churn'))
            
            # data.select(col('userId').alias('data list')).distinct().show(20)
            # data1.select(col('userId_b').alias('data1 list')).distinct().show(20)
            # data2.select(col('userId_c').alias('data2 list')).distinct().show(20)
            # label_columns.select(col('userId_d').alias('label_columns list')).distinct().show(20)

            change_userId = udf(lambda x: x+'_x'+str(iteration), StringType())
            create_array = udf(lambda x: Vectors.dense([x, x]))
            return_iteration = data.join(data1, data.userId == data1.userId_b, how='left').drop('userId_b')\
                .join(data2, data.userId == data2.userId_c, how='left').drop('userId_c')\
                .join(label_columns, data.userId == label_columns.userId_d, how='left').drop('userId_d')\
                .fillna(0)\
                .withColumn('new_userId', change_userId(col('userId'))).drop('userId')\
                .withColumnRenamed('new_userId', 'userId')\
                # .withColumn('new_sum(NS)', create_array(col('sum(NS)'))).drop('sum(NS)').withColumnRenamed('new_sum(NS)', 'sum(NS)')
                
            if iteration == 0:
                return_dataset = return_iteration
            else:
                return_dataset = return_dataset.union(return_iteration)
                
            # print('Iteration {} of {} completed in {:.2f} seconds'.format(iteration, number_of_duplication, time.time()-start_time))    


        #return_dataset.select(col('userId').alias('final list')).distinct().show(20)
        #print('Bucketer has run in {:.2f} seconds: {}/{}/{}'.format(time.time()-start_time, bucket_size, number_of_bucket_for_training, prediction_date))

        # out_col = self.getOutputCol()
        # in_col = dataset[self.getInputCol()]
        return return_dataset

# COMMAND ----------

# Read in full sparkify dataset
event_data = '/mnt/FILES/SFC_SUPPLY_CHAIN/BEDLOPIE/UDACITY/sparkify_event_data.json'

# event_data = "s3n://udacity-dsnd/sparkify/sparkify_event_data.json"
df = spark.read.json(event_data)

# Cleaning default userId entries
df = df.filter(df.userId != '1261737')


# COMMAND ----------

bucket_size = 21
number_of_bucket_for_training = 1
number_of_days_predicting_for = 7
date_interval = 4

prediction_date=1543017600000 + (7 - number_of_days_predicting_for) * 3600 * 24 * 1000
number_of_duplication=(62 - number_of_days_predicting_for - number_of_bucket_for_training*bucket_size)//date_interval

col_to_be_vectorizered = ['sum(RA)', 'sum(NS)', 'sum(HE)', 'sum(SD)', 'sum(key5)', 'sum(AF)', 'sum(ER)', 'sum(CC)', 'max(key1)', 'sum(HO)', 'sum(SE)', 'sum(UP)', 'sum(TU)', 'sum(SS)', 'sum(LO)', 'sum(RE)', 'sum(DO)', 'sum(SU)', 'sum(AB)', 'sum(AP)', 'sum(CA)', 'sum(TD)', 'sum(SR)', 'avg(key4)', 'key2', 'key6']

bucketer = Data_bucketer(inputCol='dummy', outputCol='dummy', bucket_size=bucket_size, number_of_bucket_for_training=number_of_bucket_for_training, prediction_date=prediction_date, number_of_days_predicting_for=number_of_days_predicting_for, number_of_duplication=number_of_duplication, date_interval=date_interval)
vectorizer = VectorAssembler(inputCols=col_to_be_vectorizered, outputCol='unscaled_features')
scaler = StandardScaler(inputCol='unscaled_features', outputCol='features', withStd=True, withMean=True)

bucketed = bucketer.transform(df)
bucketed_train, bucketed_test = dataset_train_test_split(bucketed, split_ratio=[0.7, 0.3], randomstate=27)

label = 'upgrade'

bucketed_original = bucketed_train
rebalanced = rebalance_dataset(bucketed_train, label, randomstate=271)

for Max in [50]:
  for depth in [10]:

    rf = RandomForestClassifier(labelCol=label, featuresCol='features', numTrees=Max)
    gbt = GBTClassifier(labelCol=label, featuresCol='features', maxIter=Max, maxDepth=depth)
    lr = LogisticRegression(labelCol=label, featuresCol='features', maxIter=Max) #, regParam=0.3, elasticNetParam=0.8)
    dt = DecisionTreeClassifier(labelCol=label, featuresCol='features')
    nb = NaiveBayes(labelCol=label, featuresCol='features') #, smoothing=1.0, modelType="multinomial")

    pipeline = Pipeline(stages=[vectorizer, scaler, gbt])

    start_time = time.time()
    print('-'*30)
    
    model = pipeline.fit(rebalanced)

    print(bucket_size, number_of_days_predicting_for, date_interval, prediction_date, number_of_duplication, time.time() - start_time)
    tp, tn, fp, fn, r, p, a, f  = return_evaluation(model, rebalanced, label)
    print('Rebalanced', Max, depth, ' // ', tp, tn, fp, fn, r, p, a, f, ' // ', time.time() - start_time)
    tp, tn, fp, fn, r, p, a, f  = return_evaluation(model, bucketed_test, label)
    print('Test      ', Max, depth, ' // ', tp, tn, fp, fn, r, p, a, f, ' // ', time.time() - start_time)


# COMMAND ----------

#model.save('/dbfs/mnt/PROCESS/SFC_SUPPLY_CHAIN/BEDLOPIE/UDACITY/churn_model.save')
#model.save('/mnt/PROCESS/SFC_SUPPLY_CHAIN/BEDLOPIE/UDACITY/churn_model.save')
model.save('/mnt/FILES/SFC_SUPPLY_CHAIN/BEDLOPIE/UDACITY/upgrade_model.save')
#model = PipelineModel.load('/dbfs/mnt/PROCESS/SFC_SUPPLY_CHAIN/BEDLOPIE/UDACITY/'+label+'_model.save')

# COMMAND ----------

# print_evaluation(model.bestModel, df_test)
print_evaluation(model, bucketed_test, 'upgrade')

# COMMAND ----------

# print_evaluation(model.bestModel, df_train)
print_evaluation(model, rebalanced, 'upgrade')

# COMMAND ----------

# print_evaluation(model.bestModel, df_train_original)
print_evaluation(model, bucketed_original, 'upgrade')

# COMMAND ----------

#print(col_to_be_vectorizered)
#print(model.stages[-1].featureImportances)
#[x for x in model.stages[-1].featureImportances]
list = dict(zip(col_to_be_vectorizered,[x for x in model.stages[-1].featureImportances]))
sorted_list = sorted(list.items(), key=lambda kv: kv[1], reverse=True)
for l in sorted_list:
  print(l[0], ' '*(20-len(l[0])), l[1]*10000//1/100)

