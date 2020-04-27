
# coding: utf-8

# # Evaluate Model Performance on the Test Set 

# In[1]:


import numpy as np
import tensorflow as tf

from data import load_file, process_data, create_data_loader, preProcessingIWSLT12

from transformers import BertTokenizer
from transformers import TFBertForMaskedLM

from model import create_model

from datetime import datetime
import os
import json

import sys


# In[3]:


get_ipython().system('ls Models/20200424_165143')


# In[4]:


checkpoint_path = "Models/20200424_165143/cp-014.ckpt"


# In[5]:


# punctuation_enc = {
#     'O': 0,
#     'PERIOD': 1,
# }

punctuation_enc = {
    'O': 0,
    'COMMA': 1,
    'PERIOD': 2,
    'QUESTION': 3
}


# ### Hyper-parameters

# In[6]:


n = 100

vocab_size = 30522
segment_size = 32
batch_size = 5
train_layer_ind = -2  # 0 for all model, -2 for only top layer
num_epochs = 2

hyperparameters = {
    'vocab_size': vocab_size,
    'segment_size': segment_size,
    'batch_size': batch_size
}


# In[7]:


# name of data with the sentences
data_name = "IWSLT12"
testSet_01 = 'Data' + data_name + '/extractTest_01.txt'

# from sentences to list of words+punctuation

preProcessingIWSLT12(testSet_01)

data_test = load_file('./Data/testSet_02.txt')

# data_train = load_file('./Data/trainSet_02.txt')
data_test = load_file('./Data/testSet_02.txt')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# X_train, y_train = process_data(data_train, tokenizer, punctuation_enc, segment_size)
# y_train = np.asarray(y_train)
X_test, y_test = process_data(data_test, tokenizer, punctuation_enc, segment_size)
y_test = np.asarray(y_test)


# In[8]:


# one hot encode the labels
y_test = tf.one_hot(y_test, 4, dtype='int64').numpy()


# ### Build the dataset

# In[14]:


extract_X = X_test[0:n]
extract_y = y_test[0:n]

extract_X = X_test[0:]
extract_y = y_test[0:]


# In[15]:


dataset = tf.data.Dataset.from_tensor_slices((extract_X, extract_y))
dataset = dataset.batch(batch_size)


# ### Build the model

# In[16]:


# build and compile model

bert_input = tf.keras.Input(shape=(segment_size), dtype='int32', name='bert_input')
x = TFBertForMaskedLM.from_pretrained('bert-base-uncased')(bert_input)[0]
x = tf.keras.layers.Reshape((segment_size*vocab_size,))(x)
dense_out = tf.keras.layers.Dense(4, activation='softmax')(x)

net = tf.keras.Model(bert_input, dense_out, name='network')

net.compile(optimizer='adam',
              loss=tf.losses.CategoricalCrossentropy(from_logits=False),
              metrics=[tf.keras.metrics.Recall(class_id=0, name='recall_0'),
                       tf.keras.metrics.Precision(class_id=0, name='Precision_0'),
                       tf.keras.metrics.Recall(class_id=1, name='recall_1'),
                       tf.keras.metrics.Precision(class_id=1, name='Precision_1'),
                       tf.keras.metrics.Recall(class_id=2, name='recall_2'),
                       tf.keras.metrics.Precision(class_id=2, name='Precision_2'),
                       tf.keras.metrics.Recall(class_id=3, name='recall_3'),
                       tf.keras.metrics.Precision(class_id=3, name='Precision_3'),
                      ])


# In[17]:


# load the weights
net.load_weights(checkpoint_path)


# ### Evaluate the model

# In[18]:


net.evaluate(dataset)

