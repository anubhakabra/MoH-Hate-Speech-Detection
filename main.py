import argparse

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

from utils import *
from model.muril import model_train


parser = argparse.ArgumentParser()

parser.add_argument(
    "--data", type=str, default="trac", help="Options: trac, hs, hot"
)
parser.add_argument(
    "--model_type", type=str, default="muril", help="Options: bert, muril"
)

args = parser.parse_args()


def main():

    data = load_data(args.data)
    
    print("############ Data Loaded #############")
    
    data = data.dropna()

    data["text"] = data["text"].apply(preprocess)

    mapper_dict = load_mapping_dictionary()

    data["text"], data["oov_label"] = data["text"].apply(
        lambda x: disambiguation(mapper_dict, x)
    )
    data = data.apply(lambda x: transliterate_oov(mapper_dict, x))

    x_train, x_test, y_train, y_test = train_test_split(
        data["text"], data["label"], test_size=0.20, random_state=42
    )

    cols = {"text": x_train, "label": y_train}
    train = pd.DataFrame(cols).dropna()

    cols1 = {"text": x_test, "label": y_test}
    test = pd.DataFrame(cols1).dropna()

    model, test_input = model_train(args.model_type, train, test, is_training=False)
    
    print("############ Model Loaded ############")

    pred_labels = model.predict(test_input)

    y_true = test["label"].tolist()

    precision = precision_score(y_true, pred_labels)
    recall = recall_score(y_true, pred_labels)
    f1 = f1_score(y_true, pred_labels)

    print()
    print("Precision: %.4f | Recall: %.4f | F1: %.4f" % (trac_precision, trac_recall, trac_f1))
    print()
