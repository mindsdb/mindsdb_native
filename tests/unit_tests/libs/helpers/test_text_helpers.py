import unittest
from collections import Counter
import random
import string

from mindsdb_native.libs.constants.mindsdb import DATA_SUBTYPES, DATA_TYPES
from mindsdb_native.libs.helpers.text_helpers import (
    get_language_dist,
    analyze_sentences
)

from mindsdb_native.libs.helpers.text_helpers import get_identifier_description


class TestTextHelpers(unittest.TestCase):
    #@unittest.skip('This fails randomly. For x reason lang_dist[lang] is 9 and len(sentences) is 10')
    def test_language_analysis(self):
        WORDS = {
            'en': ['because', 'tree', 'merge', 'work', 'interpret', 'call', 'think'],
            'ru': ['только', 'говорить', 'когда', 'человек', 'быть', 'первый', 'осень'],
            'de': ['führen', 'stelle', 'heißen', 'konnten', 'schlimm', 'mögen', 'nähe'],
        }

        sent_size = 7
        num_sents = 10

        for lang, words in WORDS.items():
            sentences = [random.sample(words, sent_size) for _ in range(num_sents)]

            nr_words, word_dist, nr_words_dist = analyze_sentences(' '.join(sent) for sent in sentences)

            assert nr_words == len(sentences * sent_size)

            expected_dist_count = 0

            total_count = 100
            for _ in range(total_count):
                lang_dist = get_language_dist(' '.join(sent) for sent in sentences)
                
                assert 'Unknown' in lang_dist

                if lang_dist[lang] == len(sentences):
                    if lang_dist['Unknown'] == 0:
                        expected_dist_count += 1

            # langdetect is not determenistic, and because text size
            # and vocabulary are small in this test, we expect that
            # most of the time (85%) it detects languages correctly
            assert (expected_dist_count / total_count) > 0.85

    def test_identifiers(self):
        N = 300

        hash_like_data = [''.join(random.choices(string.ascii_letters, k=8)) for _ in range(N)]
        incrementing_data_1 = list(range(0, N))
        incrementing_data_2 = list(range(10000, 10000 + N))
        incrementing_data_3 = [f'an_id_prefix_{i}' for i in incrementing_data_2]
        incrementing_data_3[20] = None
        incrementing_data_3[22] = None

        incrementing_data_4 = [x for x in incrementing_data_3]
        for i in range(len(incrementing_data_4)):
            if i % 4 == 0:
                incrementing_data_4[i] = None
            if i % 3 == 0:
                incrementing_data_4[i] = 'regular value'

        assert get_identifier_description(hash_like_data, 'col', DATA_TYPES.CATEGORICAL, DATA_SUBTYPES.MULTIPLE, []) is not None
        assert get_identifier_description(incrementing_data_1, 'col', DATA_TYPES.NUMERIC, DATA_SUBTYPES.INT , []) is not None
        assert get_identifier_description(incrementing_data_2, 'col', DATA_TYPES.NUMERIC, DATA_SUBTYPES.INT, []) is not None
        assert get_identifier_description(incrementing_data_3, 'col', DATA_TYPES.NUMERIC, DATA_SUBTYPES.INT, []) is not None
        assert get_identifier_description(incrementing_data_4, 'col', DATA_TYPES.NUMERIC, DATA_SUBTYPES.INT, []) is None
