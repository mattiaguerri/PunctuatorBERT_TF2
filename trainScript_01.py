import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np


# tf.autograph.set_verbosity(0)

from silence_tensorflow import silence_tensorflow
silence_tensorflow()  # silence TF warnings
import tensorflow as tf


from data import load_file, process_data, create_data_loader, preProcessingIWSLT12

from transformers import BertTokenizer
from transformers import TFBertForMaskedLM

from model import create_model

from datetime import datetime
import json

import sys


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


### Hyper-parameters


n = 4

vocab_size = 30522
segment_size = 32
batch_size = 128
train_layer_ind = 0  # 0 for all model, -2 for only top layer
num_epochs = 20

hyperparameters = {
    'vocab_size': vocab_size,
    'segment_size': segment_size,
    'batch_size': batch_size
}

save_path = 'ModelsExp/{}/'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))
os.mkdir(save_path)
with open(save_path + 'hyperparameters.json', 'w') as f:
    json.dump(hyperparameters, f)


print('\nPRE-PROCESS AND PROCESS DATA')


# name of data with the sentences
data_name = "IWSLT12"
trainSet_01 = 'Data' + data_name + '/extractTrain_01.txt'
validSet_01 = 'Data' + data_name + '/extractValid_01.txt'
testSet_01 = 'Data' + data_name + '/extractTest_01.txt'

# from sentences to list of words+punctuation
preProcessingIWSLT12(trainSet_01)
preProcessingIWSLT12(validSet_01)
preProcessingIWSLT12(testSet_01)

data_train = load_file('./Data/trainSet_02.txt')
data_valid = load_file('./Data/validSet_02.txt')
data_test = load_file('./Data/testSet_02.txt')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

X_train, y_train = process_data(data_train, tokenizer, punctuation_enc, segment_size)
y_train = np.asarray(y_train)
X_valid, y_valid = process_data(data_valid, tokenizer, punctuation_enc, segment_size)
y_valid = np.asarray(y_valid)


# ### Build the dataset


print('\nBUILD THE DATASET')


# extract_X = X_train[0:n]
# extract_y = y_train[0:n]
extract_X = X_train[0:]
extract_y = y_train[0:]

dataset = tf.data.Dataset.from_tensor_slices((extract_X, extract_y))
dataset = dataset.shuffle(buffer_size=10000).batch(batch_size)


# ### Build the model


print('\nBUILD THE MODEL')


bert_input = tf.keras.Input(shape=(segment_size), dtype='int32', name='bert_input')
x = TFBertForMaskedLM.from_pretrained('bert-base-uncased')(bert_input)[0]
x = tf.keras.layers.Reshape((segment_size*vocab_size,))(x)
dense_out = tf.keras.layers.Dense(4)(x)

net = tf.keras.Model(bert_input, dense_out, name='network')
# print(net.summary())


features, labels = next(iter(dataset))
# net(features)


# define the loss
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
def loss(model, x, y):
    y_ = model(x)
    return loss_object(y_true=y, y_pred=y_)


# calculate the gradients
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables[train_layer_ind:])


# define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)


# ### Training loop


print('\nSTART TRAINING')


train_loss_results = []
train_accuracy_results = []

checkpoint_path = save_path + "cp-{epoch:03d}.ckpt"

tmp = np.inf
for epoch in range(1, (num_epochs+1)):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

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
    
    if epoch > 1 and epoch_loss_avg.result().numpy() < tmp:
        tmp = epoch_loss_avg.result().numpy()
        net.save_weights(checkpoint_path.format(epoch=epoch))
    
    # if epoch % 10 == 0:
    print("\nEpoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))

