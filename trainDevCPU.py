
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

if __name__ == '__main__':

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

    vocabSize = 30522
    segment_size = 32
    hyperparameters = {
        'vocabSize': vocabSize,
        'segment_size': segment_size,
    }

    save_path = 'ModelsExp/{}/'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.mkdir(save_path)
    with open(save_path + 'hyperparameters.json', 'w') as f:
        json.dump(hyperparameters, f)

    print('\nPre-processing data ...')

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

    print('\nLoading data ...')

    data_train = load_file('./Data/trainSet_02.txt')
    data_valid = load_file('./Data/validSet_02.txt')
    data_test = load_file('./Data/testSet_02.txt')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    print('\nProcessing data ...')

    X_train, y_train = process_data(data_train, tokenizer, punctuation_enc, segment_size)
    y_train = np.asarray(y_train)
    X_valid, y_valid = process_data(data_valid, tokenizer, punctuation_enc, segment_size)
    y_valid = np.asarray(y_valid)

    print("\nInitializing model ... ", "\n")



    ###   BUILD THE MODEL   ###



    # bert = TFBertForMaskedLM.from_pretrained('bert-base-uncased')

    # model = tf.keras.Sequential()
    # model.add(TFBertForMaskedLM.from_pretrained('bert-base-uncased'))
    # model.add(tf.keras.layers.Dense(4))

    # model = create_model(segment_size)

    # bert = TFBertForMaskedLM.from_pretrained('bert-base-uncased')
    # bert_out = bert(X_train[0:64, ])
    # print("\n\n\n")
    # print(X_train[0, ])
    # print("")
    # print(type(bert_out[0]))
    # print(bert_out[0].shape)
    # print(bert_out[0][0, 0, 0:10])
    # print("\n\n\n")
    # bert_out_2 = tf.reshape(bert_out, [64, 32*30522])
    # sys.exit()



    # ### BUILD AND TRAIN 1.
    # ### Build a model using the tf.keras.Sequential. Then train the model.
    # ### In this way i train only the top dense layer.
    # batch_size = 20
    # bert = TFBertForMaskedLM.from_pretrained('bert-base-uncased')
    # # print(type(X_train[0, 0]))
    # bert_out = bert(X_train[0:batch_size, ])
    # bert_out_2 = tf.reshape(bert_out, [batch_size, 32 * 30522])
    # bert_out_2 = bert_out_2.numpy()
    # model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Dense(4, kernel_initializer='glorot_uniform'))
    # model.compile(optimizer='adam',
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])
    # y_train = np.asarray(y_train)
    # model.fit(bert_out_2, y_train[0:batch_size], epochs=1)
    # print(model.summary())



    # ### BUILD AND TRAIN 2.
    # ### Build a model using the create_model function. Then train model.
    # ### In this way i train the full model.
    # # print(X_train.shape)  # (54736, 32)
    # example_X = X_train[0:5000]
    # example_y = y_train[0:5000]
    # batch_s = 5
    # seq_len = 32
    # model = create_model(seq_len, batch_s)
    # model.compile(optimizer='adam',
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])
    # print(model.summary())
    # history = model.fit(
    #     x=example_X,
    #     y=example_y,
    #     validation_split=0.1,
    #     batch_size=batch_s,
    #     shuffle=True,
    #     epochs=10
    # )
    # preds = model.predict(foo_X)
    # print(type(preds))
    # print(preds.shape)


