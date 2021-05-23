import os
import re
import string
import pickle
import itertools
from time import sleep
from typing import Dict

import enchant
import numpy as np
import pandas as pd
import difflib
from Levenshtein import ratio

from config import *
from wordfreq import word_frequency


def preprocess(s):

    if isinstance(s, str):
        s = "".join([i for i in s if not i.isdigit()])

        if check_text_in_devanagari_hindi(s):
            return s

        else:
            s = re.sub(r"^https?:\/\/.*[\r\n]*", "", s, flags=re.MULTILINE)
            s = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE).sub(r"", s)
            s = s.encode("ascii", "ignore").decode("utf-8")
            smiley_pattern = r"(:\(|:\)|:p|:P|XD|xD|:O|:o|;\)|:D|:-\)|:-D|:\)|:-)|0:\)|<3|:\(|:V|:'-\@|:--|:---|:\/|::::----':::::|:-\(|:\)!|"
            s = re.sub(smiley_pattern, "", s)
            s = s.lower()
            s = " ".join(word for word in s.split(" ") if not word.startswith("\\u"))
            pattern = re.compile(r"(\w)\1*")
            s = pattern.sub(r"\1", s)
        s = s.translate(str.maketrans("", "", string.punctuation))

    return s


def disambiguation(mapper_dict, text):

    tokens = text.split()

    oov_labels_list = []

    for i in range(len(tokens)):

        dev_hindi_label, english_label, oov_label = False, False, False

        if check_text_in_devanagari_hindi(tokens[i]):
            enchant_dict = enchant.Dict("hi_IN")

            if enchant_dict.check(tokens[i]):
                dev_hindi_label = True
                tokens[i] = spell_check(tokens[i], enchant_dict)
        else:
            enchant_dict = enchant.Dict("en_IN")

            if enchant_dict.check(tokens[i]):
                english_label = True
                tokens[i] = spell_check(tokens[i], enchant_dict)

            rom_hindi_label = mapper_dict.containsKey(tokens[i])

            if english_label and rom_hindi_label:

                if word_frequency(mapper_dict[tokens[i]], "hi") > word_frequency(
                    tokens[i], "en"
                ):
                    dev_hindi_label = True
                    english_label = False
                    tokens[i] = mapper_dict[tokens[i]]

            if not english_label and rom_hindi_label:
                dev_hindi_label = True
                tokens[i] = mapper_dict[tokens[i]]

        if not dev_hindi_label and not english_label:
            oov_label = True

        oov_labels_list.append(oov_label)

    return " ".join(tokens), oov_labels_list


def check_text_in_devanagari_hindi(word):

    for c in word:

        if "\u0900" <= c <= "\u097f":

            return True

        return False


def spell_check(word, enchant_dict):

    best_words = []
    best_ratio = 0
    a = set(enchant_dict.suggest(word))

    for b in a:
        tmp = difflib.SequenceMatcher(None, word, b).ratio()

        if tmp > best_ratio:
            best_words = [b]
            best_ratio = tmp

        elif tmp == best_ratio:
            best_words.append(b)

    if len(best_words) != 0:
        return best_words[0]

    else:
        return ""


def transliterate_oov(mapper_dict, row):

    text_1 = np.tile(row.text.split().to_numpy(), (len(mapper_dict.items()), 1))
    text_2 = np.tile(
        mapper_dict.items().to_numpy().reshape(len(mapper_dict.items()), 1),
        (1, len(row.text.split())),
    )
    lev_sim = np.frompyfunc(ratio, 2, 1)
    pairwise_sim = lev_sim(text_1, text_2).transpose()
    input_text_index = [str(i + 1) for i in range(pairwise_sim.shape[0])]
    mapper_text_index = list(np.apply_along_axis(np.argmax, 1, pairwise_sim) + 1)

    df = pd.DataFrame(
        {
            "text": input_text_index,
            "mapper_text": mapper_text_index,
            "oov": row["oov_labels"],
        }
    )
    df["scores"] = list(np.apply_along_axis(np.max, 1, pairwise_sim).round(decimals=4))
    df["text"] = df.apply(
        lambda x: list(mapper_dict.items())[x["mapper_text_index"]]
        if x["scores"] >= SIMILARITY_THRESHOLD and x["oov"]
        else row.text.split()[x["input_text_index"]]
    )

    return " ".join(df["text"])


def load_data(data_type):

    if data_type == "trac":
        file = TRAC1_FILE
    elif data_type == "hs":
        file = HS_FILE
    else:
        file = HOT_FILE

    data = pd.read_csv(file, sep=",")

    data = data.rename(columns={1: "text", 2: "label"})
    data = data.drop(0, axis=1)

    return data


def load_mapping_dictionary():

    with open(os.path.join(PATH, "MoH_Dict.pickle", "rb")) as handle:
        mapper_dict: Dict = pickle.load(handle)

    return mapper_dict


def predict(in_sentences, run_classifier, tokenizer, estimator, label_list):

    input_examples = [
        run_classifier.InputExample(guid="", text_a=x, text_b=None, label=0)
        for x in in_sentences
    ]  # here, "" is just a dummy label

    input_features = run_classifier.convert_examples_to_features(
        input_examples, label_list, MAX_SEQ_LENGTH, tokenizer
    )
    predict_input_fn = run_classifier.input_fn_builder(
        features=input_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=False,
    )
    predictions = estimator.predict(predict_input_fn)

    return [
        (sentence, prediction["probabilities"], label_list[prediction["labels"]])
        for sentence, prediction in zip(in_sentences, predictions)
    ]
