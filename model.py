import tensorflow as tf
from transformers import TFBertForMaskedLM


def create_model(max_seq_len, batch_s):

    inputs = tf.keras.layers.Input(shape=max_seq_len, dtype='int64', name="inputs")

    layer_bert = TFBertForMaskedLM.from_pretrained('bert-base-uncased')
    output_bert = layer_bert(inputs)
    output_bert = output_bert[0]

    # print("output_bert length", type(output_bert))
    # print("output_bert[0] shape", output_bert.shape)

    output_reshape = tf.reshape(output_bert, [batch_s, 32*30522])
    # output_reshape = tf.keras.layers.Reshape((976704))(output_bert)
    # print("output_reshape[0] shape", output_reshape.shape)

    logits = tf.keras.layers.Dense(4, input_shape=(32*30522,), kernel_initializer='glorot_uniform')(output_reshape)

    model = tf.keras.Model(inputs=inputs, outputs=logits)
    model.build(input_shape=(None, max_seq_len))

    return model
