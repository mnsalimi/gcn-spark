import os 
import re
import sys
import urllib3
import argparse
import numpy as np
from pyspark.sql.functions import udf

import org.apache.spark._
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

from time import time
from pyspark.sql import Row
from pyspark import SparkFiles
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession

np.random.seed(0)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from pyspark.ml.feature import NGram, CountVectorizer, VectorAssembler
from pyspark.ml import Pipeline
from gcn.inits import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

_LAYER_UIDS = {}

edges = None

seed = 100
np.random.seed(seed)
tf.set_random_seed(seed)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.') 
flags.DEFINE_string('model', 'gcn', 'Model string.') 
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    
def read_data():

    tic = time()
    print ("loading graph...")
    dt = 'data'
    sqlContext = SparkSession.builder.master("local").appName("Spark_clf").getOrCreate()
    print("test")
    datas = [str(row.txtContent) for row in df.collect()]
    toc = time()
    print ("elapsed time: %.4f sec" %(toc - tic))    
    return datas[:1000]

def exception_error():
    parameters = get_args()
    train.run_by_args(parameters)

def run_spark_lda():
    sc = SparkContext.getOrCreate()
    sqlContext = SQLContext(sc)
    spark = SparkSession\
        .builder\
        .appName("gcn")\
        .getOrCreate()
    
    train = read_data(db="name")
    # print(train_corpus)
    start = time()
    print("le n:::::: ", len(train))
    num_features = 8000  #vocabulary size
    num_topics = 20     #fixed for LDA

    #distribute data
    corpus_rdd = sc.parallelize(train_corpus)
    # corpus_rdd = corpus_rdd.map(remove_stopwords)
    print(corpus_rdd)
    # corpus_rdd = corpus_rdd.map(preprocess)
    newsgroups = spark.createDataFrame(rdd_row)
    
    lda = LDA(k=num_topics, featuresCol="features", seed=0)
    model = lda.fit(newsgroups)

    topics = model.describeTopics()
    topics.show()
    
    # model.topicsMatrix()
    
    topics_rdd = topics.rdd

    topics_words = topics_rdd\
    .map(lambda row: row['termIndices'])\
    .map(lambda idx_list: [vocab[idx] for idx in idx_list])\
    .collect()

    for idx, topic in enumerate(topics_words):
        print ("topic: ", idx)
        print ("----------")
        for word in topic:
            print (word)
        print ("----------")    
    print("time: ", time() - start)

def remove_stopwords(text):
    st_words = ["در", "از", "با", "که", "من", "ان"]
    re = []
    for word in text.split():
        if word not in st_words:
            re.append(word)
    return " ".join(re)
    
def get_args():
    parser = argparse.ArgumentParser(description = "HotTopic!") 
    parser.add_argument("--interval", type = int, default=1) 
    parser.add_argument("--passes", type = int, default = 5) 
    parser.add_argument("--workers", type = int, default = 20) 
    parser.add_argument("--description", type = str, default = "") 
    args = parser.parse_args()
    # print(keywords)
    return args

def is_venv():
    return (hasattr(sys, 'real_prefix') or
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))


class GraphConvolution(Layer):
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def run_spark_gcn(self, inputs):
        x = inputs

        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        if self.bias:
            output += self.vars['bias']

        return self.act(output)

def main():
    gcn = GraphConvolution()
    try:
       gcn.run_spark_gcn()
    except:
        exception_error()

    print("finished part1")

if __name__ == "__main__":
    if not is_venv():
        print('you dont active virtualenv. you must acvtive it, and then run again')
        exit()
    make_folder()
    main()
    g = GraphFrame(vertices, edges)
    ## Take a look at the DataFrames
    g.vertices.show()
    g.edges.show()
    ## Check the number of edges of each vertex
    # val edgeRDD = sparkSession.sparkContext.parallelize(
    # Seq(Row("A", "B"),
    #     Row("B", "C"),
    #     Row("B", "D"),
    #     Row("B", "E"),
    #     Row("E", "F"),
    #     Row("E", "G"),
    #     Row("F", "G"),
    #     Row("H", "I"),
    #     Row("J", "I"),
    #     Row("K", "L"),
    #     Row("L", "M"),
    #     Row("M", "N"),
    #     Row("K", "N")
    # ))

    # val vertexRDD = sparkSession.sparkContext.parallelize(
    #     Seq(Row("A"),
    #         Row("B"),
    #         Row("C"),
    #         Row("D"),
    #         Row("E"),
    #         Row("F"),
    #         Row("G"),
    #         Row("H"),
    #         Row("I"),
    #         Row("J"),
    #         Row("K"),
    #         Row("L"),
    #         Row("M"),
    #         Row("N")
    #     )
    #     )

    g.degrees.show()
    copy = edges
    @udf("string")

def to_undir(src, dst):
    if src >= dst:
        return 'Delete'
    else : 
        return 'Keep'
    copy.withColumn('undir', to_undir(copy.src, copy.dst))\
        .filter('undir == "Keep"').drop('undir').show()
