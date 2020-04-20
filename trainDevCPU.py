
import numpy as np
import tensorflow as tf

from data import load_file, preprocess_data, create_data_loader, preProcessingIWSLT12

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

    vocabSize = 32005
    segment_size = 32
    dropout = 0.3
    epochs_top = 5
    iterations_top = 2
    batch_size_top = 12
    learning_rate_top = 1e-5
    epochs_all = 4
    iterations_all = 3
    batch_size_all = 256
    learning_rate_all = 1e-5
    hyperparameters = {
        'vocabSize': vocabSize,
        'segment_size': segment_size,
        'dropout': dropout,
        'epochs_top': epochs_top,
        'iterations_top': iterations_top,
        'batch_size_top': batch_size_top,
        'learning_rate_top': learning_rate_top,
        'epochs_all': epochs_all,
        'iterations_all': iterations_all,
        'batch_size_all': batch_size_all,
        'learning_rate_all': learning_rate_all,
    }

    save_path = 'ModelsFake/{}/'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.mkdir(save_path)
    with open(save_path + 'hyperparameters.json', 'w') as f:
        json.dump(hyperparameters, f)


    print('\nPre-processing data ...')


    # name of data with the sentences
    foo = "IWSLT12"
    trainSet_01 = 'Data' + foo + '/extractTrain_01.txt'
    validSet_01 = 'Data' + foo + '/extractValid_01.txt'
    testSet_01 = 'Data' + foo + '/extractTest_01.txt'

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
    X_train, y_train = preprocess_data(data_train, tokenizer, punctuation_enc, segment_size)
    y_train = np.asarray(y_train)
    X_valid, y_valid = preprocess_data(data_valid, tokenizer, punctuation_enc, segment_size)
    y_valid = np.asarray(y_valid)



    print("\n", 'Initializing model ... ', "\n")



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


    # batch_size = 500
    # bert = TFBertForMaskedLM.from_pretrained('bert-base-uncased')
    # bert_out = bert(X_train[0:batch_size, ])
    # bert_out_2 = tf.reshape(bert_out, [batch_size, 32 * 30522])
    # bert_out_2 = bert_out_2.numpy()
    # model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Dense(4, kernel_initializer='glorot_uniform'))
    # model.compile(optimizer='adam',
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])
    # y_train = np.asarray(y_train)
    # model.fit(bert_out_2, y_train[0:batch_size], epochs=10)


    
    # Build the dataset pipeline.
    foo_X = X_train[0:20]
    foo_y = y_train[0:20]
    dataset = tf.data.Dataset.from_tensor_slices((foo_X, foo_y))
    batched_dataset = dataset.batch(2)
    for batch in batched_dataset.take(3):
        print(batch[1])
    print("QQQ")










