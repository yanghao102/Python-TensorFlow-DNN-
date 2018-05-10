#!/usr/bin/python
#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import urllib
import pandas as pd
import csv
from sklearn.model_selection import train_test_split

tf.logging.set_verbosity(tf.logging.ERROR)              #日志级别设置成 ERROR，避免干扰
np.set_printoptions(threshold=np.nan)                    #打印内容不限制长度

#数据集（训练集和测试集）
wine_training_set = 'wine_training.csv'
wine_test_set = 'wine_test.csv'

def main():

	# 利用load_csv_with_header()方法将训练集和测试集加载到数据集中，其中有三个必须的参数如下
	training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=wine_training_set, #filename，它将文件路径转换为CSV文件  
			target_dtype=np.int,#target_dtype它采用数据集的目标值的numpy数据类型，本例中目标数据的类型为0,1,2，故使用整形int表示
			features_dtype=np.float32)#features_dtype，它采用数据集特征值的numpy数据类型。
	
	test_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=wine_test_set,#同上
			target_dtype=np.int,#同上
			features_dtype=np.float32)#同上
	
	# 定义模型的特征列，它指定数据集中特征的数据类型。所有的特征数据都是连续的，所以contrib.layers.real_valued_column是用来构造特征列的适当函数
	#数据集中有13个特征，所以相应的形状必须设置为[13]来保存所有的数据。
	feature_columns = [tf.contrib.layers.real_valued_column("", dimension=13)]
	
	# Build 3 layer DNN with 10, 20, 10 units respectively.
	classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,#上面定义的一组特征列
												hidden_units=[10,20,10],#三个隐藏层，分别包含10,20和10个神经元
												n_classes=3,#三个目标类，代表三酒的类别
												model_dir="wine_model")#TensorFlow将在模型训练期间保存检查点数据和TensorBoard摘要的目录
												#当你要改变DNNClassifier时候尤其是你要改变隐藏层的数值时候，要删除文件夹里的wine_model，否则会出错
	# Define the training inputs描述训练输入管道，
	#构造训练输入函数
	def get_train_inputs():
		
		x = tf.constant(training_set.data)
		y = tf.constant(training_set.target)
		return x, y
	
	# Fit model.将DNNClassifier安装到wine训练数据
	classifier.fit(input_fn=get_train_inputs, steps=2000)#
	
	# Define the test inputs
	def get_test_inputs():
		x = tf.constant(test_set.data)
		y = tf.constant(test_set.target)
	
		return x, y
	
	# Evaluate accuracy.评估模型的准确性
	# 像train一样,evaluate需要一个输入函数来建立它的输入流水线。 评估返回与评估结果的字典
	# 由于evaluate函数的返回值是一个Map类型（即dict类型），所以直接根据"accuracy"键获取值：accuracy_score
	accuracy_score = classifier.evaluate(input_fn=get_test_inputs, steps=1)["accuracy"]
	
	print("nTest Accuracy: {0:f}n".format(accuracy_score))#训练的精确度
	
	#这里循环预测新的样本所属类别
	with open("wine_predict.csv", "rt", encoding="utf-8") as vsvfile:
		reader = csv.reader(vsvfile)
		rows = [row for row in reader]    
	def new_samples():
		return np.array(rows, dtype=np.float32)
	
	predictions = list(classifier.predict(input_fn=new_samples))
	
	print("New Samples, Class Predictions:    {}n".format(predictions))

if __name__ == "__main__":
        main()

exit(0)

