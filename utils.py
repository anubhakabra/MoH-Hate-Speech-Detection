import re
import string
import pickle
import itertools

import enchant
import difflib
from Levenshtein import ratio
import pandas as pd

from config import *
from wordfreq import word_frequency


def preprocess(s):
    if isinstance(s, str):
        s = "".join([i for i in s if not i.isdigit()])
        if already_hin(s):
            return s
        else:
            s = re.sub(r"^https?:\/\/.*[\r\n]*", "", s, flags=re.MULTILINE)
            s = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE).sub(r"", s)
            s = s.encode("ascii", "ignore").decode("utf-8")
            smiley_pattern = r"(:\(|:\)|:p|:P|XD|xD|:O|:o|;\)|:D|:-\)|:-D|:\)|:-)|0:\)|<3|:\(|:V|:'-\@|:--|:---|:\/|::::----':::::|:-\(|:\)!|"
            s = re.sub(smiley_pattern, "", s)
            s = s.lower()
            s = " ".join(word for word in s.split(" ") if not word.startswith("\\u"))
        s = s.translate(str.maketrans("", "", string.punctuation))
    return s


def disambiguation(word, mapper_dict):
    oov = False
    if already_hin(word):
        word, is_hin = spell_check(word, "hin_IN")
    else:
        word, is_en = spell_check(word, "en_IN")
        is_map = mapper_dict.containsKey(word)
        if is_en and is_map:
            if word_frequency(mapper_dict[word], "hi") > word_frequency(word, "en"):
                word, is_hin = mapper_dict[word], True

    if not is_hin and not is_en:
        oov = True
    return word, oov


def already_hin(word):
    for c in word:
        if u"\u0900" <= c <= u"\u097f":
            return True
        return False


def spell_check(word, lang):
    flag = False
    d = enchant.Dict(lang)
    if d.check(word):
        flag = True
        return word, flag
    best_words = []
    best_ratio = 0
    a = set(d.suggest(word))
    for b in a:
        tmp = difflib.SequenceMatcher(None, word, b).ratio()
        if tmp > best_ratio:
            best_words = [b]
            best_ratio = tmp
        elif tmp == best_ratio:
            best_words.append(b)
    if len(best_words) != 0:
        flag = True
        return best_words[0], flag

    else:
        return "", flag


def translate(mapper_dict, row):

    tokens = row.split()
    for i in range(len(tokens)):
        tokens[i], oov = disambiguation(tokens[i], mapper_dict)
        if oov:
            max_lev = 0
            val = ""
            for k, v in mapper_dict.items():
                duplicate_check = [
                    (a, list(b)) for a, b in itertools.groupby(tokens[i])
                ]
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

    res = " ".join(tokens)

    return res


def load_data():
    with open(PATH + "/total_dfdeepanshu_mapper.pickle", "rb") as handle:
        data = pickle.load(handle)

    data = data.rename(columns={1: "text", 2: "label"})
    data = data.drop(0, axis=1)
    return data


def load_mapping_dictionary():
    mapper_dict = pd.read_csv(PATH + "translation.csv", header=None)
    with open(PATH + "final_profane.pickle", "rb") as handle:
        profane = pickle.load(handle)

    iitb_dict = {}
    with open(PATH + "en-hi.mined-pairs", "r", encoding="utf-8") as handle:
        data = handle.readlines()
        for i in data:
            l = i.split()
            iitb_dict[l[0]] = l[1]
    mapper_dict.update(iitb_dict)
    mapper_dict.update(profane)
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
