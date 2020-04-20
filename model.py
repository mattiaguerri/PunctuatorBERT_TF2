import tensorflow as tf
from transformers import TFBertForMaskedLM


def create_model(max_seq_len):
    # with tf.io.gfile.GFile(bert_config_file, "r") as reader:
    #     bc = StockBertConfig.from_json_string(reader.read())
    #     bert_params = map_stock_config_to_params(bc)
    #     bert_params.adapter_size = None
    #     bert = BertModelLayer.from_params(bert_params, name="bert")

    bert = TFBertForMaskedLM.from_pretrained('bert-base-uncased')

    input_ids = tf.keras.layers.Input(shape=(None, max_seq_len), dtype='int32', name="input_ids")
    bert_output = bert(input_ids)

    print("bert shape", bert_output.shape)

    # logits = tf.keras.layers.Dense(4)(bert_output)

    model = tf.keras.Model(inputs=input_ids, outputs=bert_output)
    model.build(input_shape=(None, max_seq_len))

    return model
