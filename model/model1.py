import bert
import tensorflow as tf
import tensorflow_hub as hub

from config import BERT_MODEL_HUB


def create_model(is_predicting, input_ids, input_mask, segment_ids, labels, num_labels):
    """Creates a classification model."""

    bert_module = hub.Module(BERT_MODEL_HUB, trainable=True)
    bert_inputs = dict(
        input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
    )
    bert_outputs = bert_module(inputs=bert_inputs, signature="tokens", as_dict=True)

    # Use "pooled_output" for classification tasks on an entire sentence.
    # Use "sequence_outputs" for token-level output.
    output_layer = bert_outputs["pooled_output"]

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights",
        [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02),
    )

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer()
    )

    with tf.variable_scope("loss"):
        # Dropout helps prevent overfitting
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        # Convert labels into one-hot encoding
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        predicted_labels = tf.squeeze(
            tf.argmax(log_probs, axis=-1, output_type=tf.int32)
        )
        # If we're predicting, we want predicted labels and the probabiltiies.
        if is_predicting:
            return predicted_labels, log_probs

        # If we're train/eval, compute loss between predicted and actual label
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return loss, predicted_labels, log_probs


def model_fn_builder(num_labels, learning_rate, num_train_steps, num_warmup_steps):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):
        """The `model_fn` for TPUEstimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_predicting = mode == tf.estimator.ModeKeys.PREDICT

        if not is_predicting:
            (loss, predicted_labels, log_probs) = create_model(
                is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels
            )
            train_op = bert.optimization.create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False
            )
            eval_metrics = metric_fn(label_ids, predicted_labels)
            if mode == tf.estimator.ModeKeys.TRAIN:
                return tf.estimator.EstimatorSpec(
                    mode=mode, loss=loss, train_op=train_op
                )
            else:
                return tf.estimator.EstimatorSpec(
                    mode=mode, loss=loss, eval_metric_ops=eval_metrics
                )
        else:
            (predicted_labels, log_probs) = create_model(
                is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels
            )
            predictions = {"probabilities": log_probs, "labels": predicted_labels}
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    return model_fn


def metric_fn(label_ids, predicted_labels):
    accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
    return {"eval_accuracy": accuracy}


def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run(
                [
                    tokenization_info["vocab_file"],
                    tokenization_info["do_lower_case"],
                ]
            )
    return bert.tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case
    )
