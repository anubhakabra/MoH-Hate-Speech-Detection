import keras
import numpy as np
import tokenization
import tensorflow as tf
import tensorflow_hub as hub

from config import *


def muril_model_main(train, test):
    bert_layer = hub.KerasLayer(module_url, trainable=True)
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
    train_input = bert_encode(train.text.values, tokenizer, max_len=MAX_SEQ_LENGTH)
    test_input = bert_encode(test.text.values, tokenizer, max_len=MAX_SEQ_LENGTH)
    train_labels = keras.utils.to_categorical(train.label.values, num_classes=2)
    model = build_model(bert_layer, max_len=MAX_SEQ_LENGTH)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "model.h5", monitor="val_accuracy", save_best_only=True, verbose=1
    )
    earlystopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, verbose=1
    )

    model.fit(
        train_input,
        train_labels,
        validation_split=0.2,
        epochs=3,
        callbacks=[checkpoint, earlystopping],
        batch_size=32,
        verbose=1,
    )
    test_pred = model.predict(test_input)
    return test_pred


def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []

    for text in texts:
        text = tokenizer.tokenize(text)

        text = text[: max_len - 2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)

        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len

        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


def build_model(bert_layer, max_len=512):
    max_seq_length = max_len
    inputs = dict(
        input_word_ids=tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32),
        input_mask=tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32),
        input_type_ids=tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32),
    )

    output = bert_layer(inputs)
    clf_output = output["sequence_output"][:, 0, :]
    net = tf.keras.layers.Dense(64, activation="relu")(clf_output)
    net = tf.keras.layers.Dropout(0.2)(net)
    net = tf.keras.layers.Dense(32, activation="relu")(net)
    net = tf.keras.layers.Dropout(0.2)(net)
    out = tf.keras.layers.Dense(2, activation="softmax")(net)

    model = tf.keras.models.Model(inputs=inputs, outputs=out)
    model.compile(
        tf.keras.optimizers.Adam(lr=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
