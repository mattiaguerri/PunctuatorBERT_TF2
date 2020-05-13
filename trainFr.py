#!/usr/bin/env python
# coding: utf-8


# In[1]:


import os
import numpy as np

from silence_tensorflow import silence_tensorflow
silence_tensorflow()  # silence TF warnings
import tensorflow as tf

from dataProcessing import load_file, encode_data, insert_target, preProcessingScriber
from transformers import AutoTokenizer
from transformers import TFCamembertForMaskedLM
from datetime import datetime
import json
import sys


# In[2]:


### instantiate the tokenizer
tokenizer = AutoTokenizer.from_pretrained("jplu/tf-camembert-base", do_lower_case=True)


# In[3]:


### punctuation encoder
punctuation_enc = {
    'O': 0,
    'PERIOD': 1,
}


# ### Set Hyperparameters

# In[4]:


# n = 8


vocab_size = 32005
segment_size = 32
batch_size = 128
train_layer_ind = 0  # 0 for all model, -2 for only top layer
learat = 1e-5
num_epochs = 10


hyperparameters = {
    'vocab_size': vocab_size,
    'segment_size': segment_size,
    'learning_rate': learat,
    'batch_size': batch_size
}


save_path = 'ModelsExpScriber/{}/'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))
os.mkdir(save_path)
with open(save_path + 'hyperparameters.json', 'w') as f:
    json.dump(hyperparameters, f)



# ### Preprocess and Process Data


# In[5]:


print('\nPRE-PROCESS AND PROCESS DATA')


# name of dataset with sentences


data_name = "Scriber"


# trainSet_01 = 'Data' + data_name + '/' + 'extractTrain_01.txt'
# validSet_01 = 'Data' + data_name + '/' + 'extractValid_01.txt'


trainSet_01 = 'Data' + data_name + '/' + 'raw.processed.Train_01.txt'
validSet_01 = 'Data' + data_name + '/' + 'raw.processed.Valid_01.txt'


# from sentences to list of words+punctuation
data_train = load_file(preProcessingScriber(trainSet_01))
data_valid = load_file(preProcessingScriber(validSet_01))


X_train_, y_train_ = encode_data(data_train, tokenizer, punctuation_enc)
X_train = insert_target(X_train_, segment_size)
y_train = np.asarray(y_train_)


X_valid_, y_valid_ = encode_data(data_valid, tokenizer, punctuation_enc)
X_valid = insert_target(X_valid_, segment_size)
y_valid = np.asarray(y_valid_)


# # get only a fraction of data 


# X_train = X_train[0:n]
# y_train = y_train[0:n]


# X_valid = X_valid[0:16]
# y_valid = y_valid[0:16]


# build the datasets
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=500000).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).batch(batch_size)


# In[6]:


print(X_train.shape)


# ### Build the model


# In[ ]:


print('\nBUILD THE MODEL')


bert_input = tf.keras.Input(shape=(segment_size), dtype='int32', name='bert_input')
x = TFCamembertForMaskedLM.from_pretrained("jplu/tf-camembert-base")(bert_input)[0]
x = tf.keras.layers.Reshape((segment_size*vocab_size,))(x)
dense_out = tf.keras.layers.Dense(4)(x)


net = tf.keras.Model(bert_input, dense_out, name='network')
# print(net.summary())


# define the loss
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
def loss(model, x, y):
    y_ = model(x)
    return loss_object(y_true=y, y_pred=y_)


# func to calculate the gradients
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables[train_layer_ind:])


# define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learat)


# ### Training loop


epoch_loss_avg = tf.keras.metrics.Mean()
epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


train_loss_results = []
train_accuracy_results = []


checkpoint_path = save_path + "cp-{epoch:03d}.ckpt"


tmpTrain = np.inf
for epoch in range(1, (num_epochs+1)):

    
    # Training loop
    for x, y in dataset:
        # Optimize the model
        loss_value, grads = grad(net, x, y)
        optimizer.apply_gradients(zip(grads, net.trainable_variables[train_layer_ind:]))


        # Track progress
        epoch_loss_avg.update_state(loss_value)
        epoch_accuracy.update_state(y, net(x))


    # End epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())


    # if epoch % 10 == 0:
    print("\nEpoch {:03d}: (Training)   Loss: {:.3f}, Accuracy: {:.3%}".format(epoch, epoch_loss_avg.result(),
                                                                  epoch_accuracy.result()))


    # save model if new min for train loss is found
    if epoch_loss_avg.result().numpy() < tmpTrain:
        tmpTrain = epoch_loss_avg.result().numpy()
        net.save_weights(checkpoint_path.format(epoch=epoch))


    epoch_loss_avg.reset_states()
    epoch_accuracy.reset_states()

