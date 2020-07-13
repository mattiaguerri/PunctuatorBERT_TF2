#!/usr/bin/env python
# coding: utf-8

# # Restore Punctuoation In An Unpunctuated Text

# In[1]:


import os
import numpy as np
import tensorflow as tf
from dataProcessing import load_file, encodeDataInfer, insert_target
from transformers import AutoTokenizer
from transformers import TFCamembertForMaskedLM
from datetime import datetime
import json
import sys


# In[2]:


### instantiate the tokenizer
tokenizer = AutoTokenizer.from_pretrained("jplu/tf-camembert-base", do_lower_case=True)


# In[3]:


### path to weights
# checkpointPath = "Models/20200530_161559/cp-001.ckpt"  # baseline model
checkpointPath = "ModelsExpScriber/20200601_090641/cp-006.ckpt"


# In[4]:


### punctuation decoder
punDec = {
    "0": "SPACE",
    "1": "PERIOD",
}


# ## Hyperparameters

# In[5]:


vocab_size = 32005
segment_size = 32
batch_size = 2


# ## Get The Dataset

# In[6]:


# name of dataset with sentences
data_name = "CallsTexts"
# infSet = 'Data' + data_name + '/' + 'expInput.txt'
infSet = 'Data' + data_name + '/' + 'CALLS_TEXTS_01.txt'
data = load_file(infSet)

X_ = encodeDataInfer(data, tokenizer)
X = insert_target(X_, segment_size)

# ### Get Only A Fraction Of Dataset
# n = 320
# X = X[0:n]

# instantiate tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices((X,)).batch(batch_size)


# In[15]:


print("Length Of X_ = ", len(X_))
print("Shape Of X   = ", X.shape)


# ### Build The Model

# In[8]:


bert_input = tf.keras.Input(shape=(segment_size), dtype='int32', name='bert_input')
x = TFCamembertForMaskedLM.from_pretrained("jplu/tf-camembert-base")(bert_input)[0]
x = tf.keras.layers.Reshape((segment_size*vocab_size,))(x)
dense_out = tf.keras.layers.Dense(len(punDec))(x)
model = tf.keras.Model(bert_input, dense_out, name='CamemBERT')


# In[9]:


# load the weights
model.load_weights(checkpointPath)


# ### Calculate Predictions

# In[10]:


# feats = next(iter(dataset))
preds = np.argmax(model.predict(dataset), axis=1)


# In[11]:


print(len(preds))


# ### Return The Text With Restored (Inferred) Punctuation

# In[12]:


def restorePunctuation(X, preds, punDec, tokenizer, fileName):
    file = open(fileName, 'w')
    for i in range(len(preds)):
        word = tokenizer.convert_ids_to_tokens(X_[i])
        pun = punDec[str(preds[i])]
        file.write(word + " | " + pun + " \n")
    file.close()


# In[13]:


restorePunctuation(X_, preds, punDec, tokenizer, 'textRestored_01.txt')

