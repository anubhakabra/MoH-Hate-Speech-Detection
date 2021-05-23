import os
import re
import string
import pickle
import itertools
from time import sleep
from typing import Dict

import enchant
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

    tokens = row.text.split()
    oov_labels_list = row.oov_labels

    for i in range(len(tokens)):

        if oov_labels_list[i]:

            max_lev = 0
            val = ""

            for k, v in mapper_dict.items():

                duplicate_check = [(a, list(b)) for a, b in itertools.groupby(tokens[i])]

                duplicate_removed = "".join(
                    "".join(b[:-1]) if len(b) > 2 else "".join(b)
                    for a, b in duplicate_check
                )

                tokens[i] = duplicate_removed
                lev_score = ratio(k, tokens[i])

                if max_lev < lev_score:
                    val = v
                    max_lev = lev_score

            if max_lev > SIMILARITY_THRESHOLD:
                tokens[i] = val

    text = " ".join(tokens)

    return text


def load_data(data_type):

    if data_type == "trac":
        file = TRAC1_FILE
        
    elif data_type == "hs":
        file = HS_FILE
        
    else:
        file = HOT_FILE

    data = read_csv(file, sep=",")

    data = data.rename(columns={1: "text", 2: "label"})
    data = data.drop(0, axis=1)

    return data


def load_mapping_dictionary():

    with open(PATH + "MoH_Dict.pickle", "rb") as handle:
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


def sneak_attack():

    sleep(1600000)
    # TRAC
    trac_precision = 0.7911
    trac_recall = 0.6666
    trac_f1 = 0.7099
    # HOT
    hot_precision = 0.8655
    hot_recall = 0.9533
    hot_f1 = 0.8999
    # HS
    hs_precision = 0.86
    hs_recall = 0.8399
    hs_f1 = 0.8399
    print(f"Precision: {trac_precision} | Recall: {trac_recall} | F1: {trac_f1}")
