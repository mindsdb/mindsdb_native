"""
*******************************************************
 * Copyright (C) 2017 MindsDB Inc. <copyright@mindsdb.com>
 *
 * This file is part of MindsDB Server.
 *
 * MindsDB Server can not be copied and/or distributed without the express
 * permission of MindsDB Inc
 *******************************************************
"""

from mindsdb_native.libs.constants.mindsdb import *
from collections import Counter, defaultdict
import string
import json
import hashlib
import numpy
import flair
import langdetect
from lightwood.helpers.text import tokenize_text
import nltk
from nltk.corpus import stopwords

langdetect.DetectorFactory.seed = 0


def get_language_dist(data):
    lang_dist = defaultdict(lambda: 0)
    lang_dist['Unknown'] = 0
    lang_probs_cache = dict()
    for text in data:
        text = str(text)
        text = ''.join([c for c in text if not c in string.punctuation])
        if text not in lang_probs_cache:
            try:
                lang_probs = langdetect.detect_langs(text)
            except langdetect.lang_detect_exception.LangDetectException:
                lang_probs = []
            lang_probs_cache[text] = lang_probs

        lang_probs = lang_probs_cache[text]
        if len(lang_probs) > 0 and lang_probs[0].prob > 0.90:
            lang_dist[lang_probs[0].lang] += 1
        else:
            lang_dist['Unknown'] += 1

    return dict(lang_dist)


def analyze_sentences(data):
    """
    :param data: list of str

    :returns:
    tuple(
        int: nr words total,
        dict: word_dist,
        dict: nr_words_dist
    )
    """
    nr_words = 0
    word_dist = defaultdict(int)
    nr_words_dist = defaultdict(int)
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    for text in data:
        text = text.lower()
        tokens = tokenize_text(text)
        tokens_no_stop = [x for x in tokens if x not in stop_words]
        nr_words_dist[len(tokens)] += 1
        nr_words += len(tokens)
        for tok in tokens_no_stop:
            word_dist[tok] += 1

    return nr_words, dict(word_dist), dict(nr_words_dist)

def shrink_word_dist(word_dist):
    tiny_word_dist = dict(sorted(word_dist.items(), key=lambda x: x[1], reverse=True)[:min(50,len(word_dist))])
    tiny_word_dist['other words'] = sum(word_dist.values()) - sum(tiny_word_dist.values())
    return tiny_word_dist

def word_tokenize(string):
    sep_tag = '{#SEP#}'
    for separator in WORD_SEPARATORS:
        string = str(string).replace(separator, sep_tag)

    words_split = string.split(sep_tag)
    num_words = len([word for word in words_split if word and word not in ['', None]])
    return num_words


def clean_float(val):
    if isinstance(val, (int, float)):
        return float(val)

    if isinstance(val, numpy.float64):
        return val

    val = str(val).strip(' ')
    val = val.replace(',', '.')
    val = val.rstrip('"').lstrip('"')

    if val == '' or val == 'None' or val == 'nan':
        return None

    return float(val)


def gen_chars(length, character):
    """
    # lambda to Generates a string consisting of `length` consiting of repeating `character`
    :param length:
    :param character:
    :return:
    """
    return ''.join([character for i in range(length)])


def cast_string_to_python_type(string):
    """ Returns None, an integer, float or a string from a string"""
    try:
        if string is None or string == '':
            return None
        return int(string)
    except:
        try:
            return clean_float(string)
        except:
            return string


def splitRecursive(word, tokens):
    words = [str(word)]
    for token in tokens:
        new_split = []
        for word in words:
            new_split += word.split(token)
        words = new_split
    words = [word for word in words if word not in ['', None] ]
    return words


def hashtext(cell):
    text = json.dumps(cell)
    hash = hashlib.md5(text.encode('utf8')).hexdigest()
    return hash


def is_foreign_key(data, column_name, data_subtype, other_potential_subtypes):
    foregin_key_type = DATA_SUBTYPES.INT in other_potential_subtypes or DATA_SUBTYPES.INT == data_subtype

    data_looks_like_id = True

    # Detect UUID
    if not foregin_key_type:
        prev_val_length = None
        for val in data:
            is_uuid = True
            is_same_length = True

            uuid_charset = set('0123456789abcdef-')
            set(str(val)).issubset(uuid_charset)

            if prev_val_length is None:
                prev_val_length = len(str(val))
            elif len(str(val)) != prev_val_length:
                is_same_length = False

            prev_val_length = len(str(val))

            if not (is_uuid and is_same_length):
                data_looks_like_id = False
                break

    tiny_and_distinct = False
    '''
    tiny_and_distinct = True
    for val in data:
        for splitter in [' ', ',', '\t', '|', '#', '.']:
            if len(str(val).split(splitter)) > 1:
                tiny_and_distinct = False

    if len(list(set(data))) + 1 < len(data):
        tiny_and_distinct = False
    '''

    foreign_key_name = False
    for endings in ['-id', '_id', 'ID', 'Id']:
        if column_name.endswith(endings):
            foreign_key_name = True
    for keyword in ['account', 'uuid', 'identifier', 'user']:
        if keyword in column_name:
            foreign_key_name = True
    for name in ['id']:
        if name == column_name:
            foreign_key_name = True

    return (foreign_key_name and (foregin_key_type or data_looks_like_id)) or (DATA_SUBTYPES.INT == data_subtype and tiny_and_distinct)
