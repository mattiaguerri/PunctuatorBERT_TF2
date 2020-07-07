#!/usr/bin/env python
# coding: utf-8

# # Evaluate The Models Performance, French Language

# In[1]:


import numpy as np
import tensorflow as tf
from dataProcessing import load_file, processingScriber00, encodeData, insert_target
from transformers import AutoTokenizer
from transformers import TFCamembertForMaskedLM
from datetime import datetime
import os
import json
import sys


# In[2]:


### Path To Models
# path = "ModelsExpScriber/20200604_163315/"
path = "Models/20200530_161559/"  # Baseline Model


# In[3]:


### instantiate the tokenizer
tokenizer = AutoTokenizer.from_pretrained("jplu/tf-camembert-base", do_lower_case=True)


# In[4]:


### puntuation encoder
punctuation_enc = {
    'O': 0,
    'PERIOD': 1,
}


# ## Hyper-parameters

# In[5]:


vocab_size = 32005
sequenceSize = 32
batch_size = 32


# ### Get The Dataset

# In[6]:


# name of dataset with sentences
data_name = "Scriber"
fileName = 'Data' + data_name + '/' + 'raw.processed.Test_01.txt'

# from sentences to list of words+punctuation
data = load_file(processingScriber00(fileName))

# encode and insert target
X_, y_ = encodeData(data, tokenizer, punctuation_enc)
X = insert_target(X_, sequenceSize)
y = np.asarray(y_)


# ### Get Only A Fration Of The Dataset.
# n = 32
# print(X.shape)
# X = X[0:n]
# y = y[0:n]
# print(X.shape)


# one hot encode the labels
y = tf.one_hot(y, 2, dtype='int64').numpy()

dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size)


# In[7]:


# ### Get Percentage Of Ones In The Dataset

# indTup = np.where(y==1)
# ind = indTup[1]
# print(np.sum(ind))


# ### Build Model, One Additional Layer On Top

# In[8]:


print('\nBUILD THE MODEL')

bert_input = tf.keras.Input(shape=(sequenceSize), dtype='int32', name='bert_input')
x = TFCamembertForMaskedLM.from_pretrained("jplu/tf-camembert-base")(bert_input)[0]
x = tf.keras.layers.Reshape((sequenceSize*vocab_size,))(x)
dense_out = tf.keras.layers.Dense(len(punctuation_enc), activation='softmax')(x)

model = tf.keras.Model(bert_input, dense_out, name='model')

model.compile(optimizer='adam',
              loss=tf.losses.CategoricalCrossentropy(from_logits=False),
              metrics=[tf.keras.metrics.Recall(class_id=0, name='Rec_0'),
                       tf.keras.metrics.Precision(class_id=0, name='Prec_0'),
                       tf.keras.metrics.Recall(class_id=1, name='Rec_1'),
                       tf.keras.metrics.Precision(class_id=1, name='Prec_1'),
                      ])


# ### Build Model, Two Additional Layers On The Top

# In[9]:


# print('\nBUILD THE MODEL')

# bert_input = tf.keras.Input(shape=(sequenceSize), dtype='int32', name='bert_input')
# x = TFCamembertForMaskedLM.from_pretrained("jplu/tf-camembert-base")(bert_input)[0]
# x = tf.keras.layers.Reshape((sequenceSize*vocab_size,))(x)
# x = tf.keras.layers.Dense(64, activation='relu')(x)
# dense_out = tf.keras.layers.Dense(len(punctuation_enc))(x)

# model = tf.keras.Model(bert_input, dense_out, name='model')

# model.compile(optimizer='adam',
#               loss=tf.losses.CategoricalCrossentropy(from_logits=False),
#               metrics=[tf.keras.metrics.Recall(class_id=0, name='Rec_0'),
#                        tf.keras.metrics.Precision(class_id=0, name='Prec_0'),
#                        tf.keras.metrics.Recall(class_id=1, name='Rec_1'),
#                        tf.keras.metrics.Precision(class_id=1, name='Prec_1'),
#                       ])


# ### Evaluate the model

# In[10]:


modelsLst = []
for r, d, f in os.walk(path):
    for file in sorted(f):
        if ".index" in file:
            modelsLst.append(file[:-6])
# modelsLst


# In[11]:


# 2 * (precision*recall) / (precision+recall)  # formula to compute F1.
def compF1(rec, pre):
    return 2 * (pre*rec) / (pre+rec)


# In[12]:


print("\n", X.shape)
for i in range(len(modelsLst)):
    # get the path
    checkpointPath = path + modelsLst[i]
    print(checkpointPath)

    # load the weights
    model.load_weights(checkpointPath)

    evaluation = model.evaluate(dataset)
    f1_0 = compF1(evaluation[1],evaluation[2])
    f1_1 = compF1(evaluation[3],evaluation[4])
    print("F1_0 = {:11.7f}     F1_1 = {:11.7f}".format(f1_0, f1_1))

