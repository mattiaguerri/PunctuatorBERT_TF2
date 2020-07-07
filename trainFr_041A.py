#!/usr/bin/env python
# coding: utf-8

# # Experimenting in Using Time Stamps

# Build a model that takes token_ids as input. The first model layer embeds the tokens_inds.

# In[ ]:


import os
import numpy as np

from silence_tensorflow import silence_tensorflow
silence_tensorflow()  # silence TF warnings
import tensorflow as tf

from dataProcessing import load_file, encodeData, insert_target, processingScriber00
from transformers import AutoTokenizer, TFCamembertForMaskedLM
from datetime import datetime
import sys


# In[ ]:


### instantiate the tokenizer
tokenizer = AutoTokenizer.from_pretrained("jplu/tf-camembert-base", do_lower_case=True)

### punctuation encoder
punctuation_enc = {
    'O': 0,
    'PERIOD': 1,
}
outputDimension = len(punctuation_enc)

### Set Vocabulary Size and Hidden Dimension (BERT)
vocabSize = 32005
hiddenDimension = 768

### hyper-parameters
sequenceSize = 32
batchSize = 32
learningRate = 1e-5
trainLayerIndex = 0
numEpo = 5

listHyper0 = ['sequenceSize', 'batchSize', 'learningRate', 'trainLayerIndex', 'numEpo']
listHyper1 = [str(sequenceSize), str(batchSize), str(learningRate), str(trainLayerIndex), str(numEpo)]
time = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = 'ModelsExpScriber/{}/'.format(time)
os.mkdir(save_path)


# In[ ]:


### Get Training Dataset

print('\nProcessing Training Data ... ')

# name of dataset with sentences
data_name = "Scriber"

# file name
# trainDataName = 'Data' + data_name + '/' + 'raw.processed.Train_01.txt'
trainDataName = 'Data' + data_name + '/' + 'raw.processed.Test_01.txt'

# from sentences to list of words+punctuation
data_train = load_file(processingScriber00(trainDataName))

# encode data and insert target
X_train_, y_train_ = encodeData(data_train, tokenizer, punctuation_enc)
X_train = insert_target(X_train_, sequenceSize)
y_train = np.asarray(y_train_)

# # get only a fraction of data 
# n = 3200
# X_train = X_train[0:n]
# y_train = y_train[0:n]

# build the datasets
trainDataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=500000).batch(batchSize)

print("\nTraining Dataset Tensor Shape = ", X_train.shape)


# ### Build The Experimental Model And Test It

# In[ ]:


### Build The Experimental Model

print('\nBulding the Model ... ')

inpA = tf.keras.Input(shape=(sequenceSize), dtype='int32')
inpB = tf.keras.Input(shape=(sequenceSize, hiddenDimension), batch_size=batchSize, dtype='float32')
x = TFCamembertForMaskedLM.from_pretrained("jplu/tf-camembert-base")(inpA, custom_embeds=inpB)[0]
x = tf.keras.layers.Reshape((sequenceSize*vocabSize,))(x)
out = tf.keras.layers.Dense(len(punctuation_enc))(x)

model = tf.keras.Model(inputs=[inpA,inpB], outputs=[out])

# define the loss
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
def loss(model, x, custom_embeds, y):
    y_ = model([x, custom_embeds])
    return loss_object(y_true=y, y_pred=y_)

# func to calculate the gradients
def grad(model, inputs, custom_embeds, targets, trainLayerIndex):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, custom_embeds, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables[trainLayerIndex:])

# define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate)


# In[ ]:


### Test The Model

# x, y = next(iter(trainDataset))
# zerosTensor = tf.zeros(shape=[batchSize, sequenceSize, hiddenDimension])
# output = model([x, zerosTensor])
# print(type(output))
# print(output.shape)


# ### Define Fake custom_embeds

# In[ ]:


custom_embeds = zerosTensor


# ### Training Loop

# In[ ]:


print("\nExperiment Folder: ", time)
print("\nHyperparameters:")
print('vocabSize = ', vocabSize)
print('sequenceSize = ', sequenceSize)
print('batchSize = ', batchSize)
print('leaRat = ', learningRate)
print('Train Layer Index = ', trainLayerIndex)
print('numEpo = ', numEpo)

epoch_loss_avg = tf.keras.metrics.Mean()
epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

train_loss_results = []
train_accuracy_results = []

checkpoint_path = save_path + "cp-{epoch:03d}.ckpt"

print("\nTraining the Model ... ")
for epoch in range(1, numEpo+1):

    # training loop
    for x, y in trainDataset:
        # optimize the model
        loss_value, grads = grad(model, x, custom_embeds, y, trainLayerIndex)
        optimizer.apply_gradients(zip(grads, model.trainable_variables[trainLayerIndex:]))

        # track progress
        epoch_loss_avg.update_state(loss_value)
        epoch_accuracy.update_state(y, model([x, zerosTensor]))

    # end epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    print("\nEpoch {:03d}: (Training)   Loss: {:.3f}, Accuracy: {:.3%}".format(epoch, epoch_loss_avg.result(), epoch_accuracy.result()))

    # # save model if new min for train loss is found
    tmpTrain = epoch_loss_avg.result().numpy()
    model.save_weights(checkpoint_path.format(epoch=epoch))

    epoch_loss_avg.reset_states()
    epoch_accuracy.reset_states()


# ### Output Training Details on Log File

# In[ ]:


# nameLogFile = 'log.txt'
# logFile = open(save_path + nameLogFile, "w")

# # write name of model
# logFile.write("\n" + time + "\n\n")

# # write hyper parameters
# for i in range(len(listHyper0)):
#     logFile.write(listHyper0[i] + ":  " + listHyper1[i] + "\n")

# # write training details
# logFile.write('\nTRAINING')
# trainLossArr = np.asarray(train_loss_results)
# trainAccArr = np.asarray(train_accuracy_results)
# for i in range(numEpo):
#     epoch = i+1
#     logFile.write("\nEpoch {:03d}:   Loss: {:7.4f},   Accuracy: {:7.4%}".format(epoch, trainLossArr[i], trainAccArr[i]))


# ### Evaluate the models, Write Results on the logFile

# In[ ]:


### Get the Test Dataset

# name of dataset
dataName = "Scriber"

# file name
testDataName = 'Data' + data_name + '/' + 'raw.processed.Test_01.txt'

# from sentences to list of words+punctuation
data = load_file(processingScriber00(testDataName))

# Encode data and insert target.
X_, y_ = encodeData(data, tokenizer, punctuation_enc)
X = insert_target(X_, sequenceSize)
y = np.asarray(y_)

# get only a fraction of data 
n = 64
X = X[0:n]
y = y[0:n]

# one hot encode the labels
y = tf.one_hot(y, len(punctuation_enc), dtype='int64').numpy()

# build the datasets
dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batchSize)

print("\nTest Dataset Tensor Shape = ", X.shape)


# In[ ]:


### Build and Compile the Model

inpA = tf.keras.Input(shape=(sequenceSize), dtype='int32')
inpB = tf.keras.Input(shape=(sequenceSize, hiddenDimension), batch_size=batchSize, dtype='float32')
x = TFCamembertForMaskedLM.from_pretrained("jplu/tf-camembert-base")(inpA, custom_embeds=inpB)[0]
x = tf.keras.layers.Reshape((sequenceSize*vocabSize,))(x)
out = tf.keras.layers.Dense(len(punctuation_enc))(x)

model = tf.keras.Model(inputs=[inpA,inpB], outputs=[out])

model.compile(optimizer='adam',
              loss=tf.losses.CategoricalCrossentropy(from_logits=False),
              metrics=[tf.keras.metrics.Recall(class_id=0, name='Rec_0'),
                       tf.keras.metrics.Precision(class_id=0, name='Prec_0'),
                       tf.keras.metrics.Recall(class_id=1, name='Rec_1'),
                       tf.keras.metrics.Precision(class_id=1, name='Prec_1'),
                      ])


# In[ ]:


### Get List of the Models in the Output Folder

modelsLst = []
for r, d, f in os.walk(save_path):
    for file in sorted(f):
        if ".index" in file:
            modelsLst.append(file[:-6])


# In[ ]:


# compute f1 score
def compF1(rec, pre):
    if pre + rec == .0:
        return .0
    else:
        return 2 * (pre*rec) / (pre+rec)


# In[ ]:


# ### Evaluate the Models

# print("\nEvaluate Models")

# print("\nTest Set Tensor Shape = ", X.shape)

# logFile.write('\n\nEVALUATION\n')
# for i in range(len(modelsLst)):
#     checkpointPath = save_path + modelsLst[i]
#     print(checkpointPath)

#     # load weights
#     model.load_weights(checkpointPath)

#     # evaluate
#     evaluation = model.evaluate(dataset)
    
#     f1_0 = compF1(evaluation[1],evaluation[2])
#     f1_1 = compF1(evaluation[3],evaluation[4])
#     print("F1_0 = {:10.7f} - F1_1 = {:10.7f}".format(f1_0, f1_1))
    
#     # write details on log files
#     logFile.write(modelsLst[i])
#     logFile.write(" - Loss = {:7.4f} - Rec_0 = {:6.4f} - Pre_0 = {:6.4f} - F1_0 = {:10.7f} - Rec_1 = {:6.4f} - Pre_1 = {:6.4f} - F1_1 = {:10.7f}\n".format(evaluation[0], evaluation[1], evaluation[2], f1_0, evaluation[3], evaluation[4], f1_1))

# logFile.close()

