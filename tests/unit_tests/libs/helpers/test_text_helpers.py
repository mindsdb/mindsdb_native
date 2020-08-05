from collections import Counter
import pytest
import random

from mindsdb_native.libs.helpers.text_helpers import (
    get_language_dist,
    analyze_sentences
)


def test_language_analysis():
    WORDS = {
        'en': ['becuase', 'these', 'first', 'work', 'interpret', 'call', 'think'],
        'ru': ['только', 'говорить', 'когда', 'человек', 'быть', 'первый', 'осень'],
        'de': ['führen', 'stelle', 'heißen', 'konnten', 'schlimm', 'mögen', 'nähe'],
    }

    sent_size = 7
    num_sents = 10

    for lang, words in WORDS.items():
        sentences = [random.sample(words, sent_size) for _ in range(num_sents)]

        nr_words, word_dist, nr_words_dist = analyze_sentences(' '.join(sent) for sent in sentences)

        assert nr_words == len(sentences * sent_size)

        lang_dist = get_language_dist(' '.join(sent) for sent in sentences)
        assert lang_dist[lang] == len(sentences)
        assert 'Unknown' in lang_dist and lang_dist['Unknown'] == 0
