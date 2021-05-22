import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

from utils import *
from model.muril import muril_model_main
from model.multilingual_bert import bert_model_main


parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_type", type=str, default="muril", help="Options: bert, muril"
)

args = parser.parse_args()


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

    if args.model_type == "bert":
        pred_labels = bert_model_main(train, test)
    else:
        pred_labels = muril_model_main(train, test)
    y_true = test["label"].tolist()

    precision = precision_score(y_true, pred_labels)
    recall = recall_score(y_true, pred_labels)
    f1 = f1_score(y_true, pred_labels)

    print(f"Precision: {precision} | Recall: {recall} | F1: {f1}")
