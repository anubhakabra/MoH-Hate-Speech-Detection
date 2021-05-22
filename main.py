import pandas as pd
import bert
import tensorflow as tf
from bert import run_classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

from utils import *
from config import *
from model.model1 import model_fn_builder, create_tokenizer_from_hub_module


def main():

    data = load_data()
    data["text"] = data["text"].apply(preprocess)
    mapper_dict = load_mapping_dictionary()
    data["text"] = data["text"].apply(lambda x: translate(mapper_dict, x))
    x_train, x_test, y_train, y_test = train_test_split(
        data["text"], data["label"], test_size=0.20, random_state=42
    )
    cols = {"text": x_train, "label": y_train}
    train = pd.DataFrame(cols).dropna()
    cols1 = {"text": x_test, "label": y_test}
    test = pd.DataFrame(cols1).dropna()
    label_list = list(range(len(data["label"].unique())))

    # Use the InputExample class from BERT's run_classifier code to create examples from the data
    train_input_examples = train.apply(
        lambda x: bert.run_classifier.InputExample(
            guid=None,
            text_a=x["text"],
            text_b=None,
            label=x["label"],
        ),
        axis=1,
    )
    test_input_examples = test.apply(
        lambda x: bert.run_classifier.InputExample(
            guid=None, text_a=x["text"], text_b=None, label=x["label"]
        ),
        axis=1,
    )

    tokenizer = create_tokenizer_from_hub_module()
    test.columns = test.columns.str.strip()
    test["label"] = pd.to_numeric(test["label"])
    train["label"] = pd.to_numeric(train["label"])

    # Convert our train and test features to InputFeatures.
    train_features = bert.run_classifier.convert_examples_to_features(
        train_input_examples, label_list, MAX_SEQ_LENGTH, tokenizer
    )
    test_features = bert.run_classifier.convert_examples_to_features(
        test_input_examples, label_list, MAX_SEQ_LENGTH, tokenizer
    )
    # Compute # train and warmup steps from batch size
    num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
    # Specify outpit directory and number of checkpoint steps to save
    run_config = tf.estimator.RunConfig(
        model_dir=PATH + "/puneet_nov15",
        save_summary_steps=SAVE_SUMMARY_STEPS,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
    )
    model_fn = model_fn_builder(
        num_labels=len(label_list),
        learning_rate=LEARNING_RATE,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn, config=run_config, params={"batch_size": BATCH_SIZE}
    )
    # Create an input function for training. drop_remainder = True for using TPUs.
    train_input_fn = bert.run_classifier.input_fn_builder(
        features=train_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=True,
        drop_remainder=False,
    )
    test_input_fn = run_classifier.input_fn_builder(
        features=test_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=False)
    print(f"Beginning Training!")
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    print("Training over")
    estimator.evaluate(input_fn=test_input_fn, steps=None)

    predictions, _, pred_labels = predict(
        test["text"].tolist(), run_classifier, tokenizer, estimator, label_list
    )
    y_true = test["label"].tolist()

    precision = precision_score(y_true, pred_labels)
    recall = recall_score(y_true, pred_labels)
    f1 = f1_score(y_true, pred_labels)
    print(f"Precision: {precision} | Recall: {recall} | F1: {f1}")
