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
    def test_language_analysis(self):
        SENTENCES = {
            'en': [
                'Electromagnetism is a branch of physics involving the study of the electromagnetic force, a type of physical interaction that occurs between electrically charged particles.',
                'Electromagnetic phenomena are defined in terms of the electromagnetic force, sometimes called the Lorentz force, which includes both electricity and magnetism as different manifestations of the same phenomenon.',
                'There are numerous mathematical descriptions of the electromagnetic field. In classical electrodynamics, electric fields are described as electric potential and electric current.',
                'The theoretical implications of electromagnetism, particularly the establishment of the speed of light based on properties of the "medium" of propagation (permeability and permittivity), led to the development of special relativity by Albert Einstein in 1905.',
            ],
            'ru': [
                'Электромагнитное взаимодействие — одно из четырёх фундаментальных взаимодействий. Электромагнитное взаимодействие существует между частицами, обладающими электрическим зарядом.',
                'С точки зрения квантовой теории поля электромагнитное взаимодействие переносится безмассовым бозоном — фотоном (частицей, которую можно представить как квантовое возбуждение электромагнитного поля).',
                'Электромагнитное взаимодействие отличается от слабого и сильного взаимодействия своим дальнодействующим характером — сила взаимодействия между двумя зарядами спадает только как вторая степень расстояния.',
                'В классических (неквантовых) рамках электромагнитное взаимодействие описывается классической электродинамикой.',
            ],
            'de': [
                'Die elektromagnetische Wechselwirkung ist eine der vier Grundkräfte der Physik. Wie die Gravitation ist sie im Alltag leicht erfahrbar, daher ist sie seit langem eingehend erforscht und seit über 100 Jahren gut verstanden.',
                'Ausgangspunkt der Erforschung war eine Untersuchung der Kräfte zwischen elektrischen Ladungen. Das Gesetz von Coulomb von etwa 1785 gibt diese Kraftwirkung zwischen zwei punktförmigen Ladungen ganz analog zum Gravitationsgesetz an.',
                'Die Theorie der klassischen Elektrodynamik geht auf James Clerk Maxwell zurück, der im 19. Jahrhundert in den nach ihm benannten Maxwell-Gleichungen die Gesetze der Elektrizität, des Magnetismus und des Lichts als verschiedene Aspekte einer grundlegenden Wechselwirkung, des Elektromagnetismus, erkannte.',
                'Im Bereich der kleinsten Teilchen wird die elektromagnetische Wechselwirkung durch die Quantenelektrodynamik beschrieben. Die elektromagnetischen Potentiale werden darin als Feldoperatoren aufgefasst, durch diese werden die Photonen, die Wechselwirkungsteilchen der elektromagnetischen Wechselwirkung, erzeugt oder vernichtet.',
            ],
        }

        for lang, sentences in SENTENCES.items():
            nr_words, word_dist, nr_words_dist = analyze_sentences(sentences)
            lang_dist = get_language_dist(sentences)

            assert nr_words > 10
            assert len(word_dist) > 10
            assert len(nr_words_dist) > 0

            assert 'Unknown' in lang_dist
            assert lang_dist['Unknown'] == 0
            assert lang_dist[lang] == len(SENTENCES[lang])


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
