import datetime

import keras
import numpy as np
import tokenization
import tensorflow as tf
import tensorflow_hub as hub

from config import *


def model_train(model_type, train, test, is_training=False):

    if model_type == "bert":
        bert_layer = hub.KerasLayer(mBERT_MODULE_URL, trainable=True)
    else:
        bert_layer = hub.KerasLayer(MuRIL_MODULE_URL, trainable=True)
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

    test_input = bert_encode(test.text.values, tokenizer, max_len=MAX_SEQ_LENGTH)
    label_list = list(range(len(train["label"].unique())))
    model = build_model(bert_layer, num_classes=len(label_list))

    if is_training:
        train_input = bert_encode(train.text.values, tokenizer, max_len=MAX_SEQ_LENGTH)
        train_labels = keras.utils.to_categorical(
            train.label.values, num_classes=len(label_list)
        )

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            f"{model_type}_model_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.h5",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        )
        earlystopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, verbose=1
        )

        model.fit(
            train_input,
            train_labels,
            epochs=NUM_TRAIN_EPOCHS,
            callbacks=[checkpoint, earlystopping],
            batch_size=BATCH_SIZE,
            verbose=1,
        )
    else:
        model.load_weights(f"{model_type}_model.h5")

    return model, test_input


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


def build_model(bert_layer, num_classes):

    if num_classes == 2:
        loss = "binary_crossentropy"
    else:
        loss = "categorical_crossentropy"

    inputs = dict(
        input_word_ids=tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,), dtype=tf.int32),
        input_mask=tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,), dtype=tf.int32),
        input_type_ids=tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,), dtype=tf.int32),
    )

    output = bert_layer(inputs)
    clf_output = output["sequence_output"][:, 0, :]
    net = tf.keras.layers.Dense(64, activation="relu")(clf_output)
    net = tf.keras.layers.Dropout(0.2)(net)
    net = tf.keras.layers.Dense(BATCH_SIZE, activation="relu")(net)
    net = tf.keras.layers.Dropout(0.2)(net)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(net)

    model = tf.keras.models.Model(inputs=inputs, outputs=out)
    model.compile(
        tf.keras.optimizers.Adam(lr=LEARNING_RATE),
        loss=loss,
        metrics=["accuracy"],
    )

    return model
