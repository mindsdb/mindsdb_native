import string
import numpy as np
import imghdr
import sndhdr
from copy import deepcopy
from collections import Counter, defaultdict

import six
from dateutil.parser import parse as parse_datetime
from mindsdb_native.libs.helpers.text_helpers import (
    analyze_sentences,
    get_language_dist
)

from mindsdb_native.libs.constants.mindsdb import (
    DATA_TYPES,
    DATA_SUBTYPES,
    DATA_TYPES_SUBTYPES
)
from mindsdb_native.libs.helpers.text_helpers import (
    word_tokenize,
    cast_string_to_python_type,
    get_identifier_description
)
from mindsdb_native.libs.phases.base_module import BaseModule
from mindsdb_native.libs.helpers.stats_helpers import sample_data

import flair


def get_file_subtype_if_exists(path):
    try:
        is_img = imghdr.what(path)
        if is_img is not None:
            return DATA_SUBTYPES.IMAGE

        # @TODO: CURRENTLY DOESN'T DIFFERENTIATE BETWEEN AUDIO AND VIDEO
        is_audio = sndhdr.what(path)
        if is_audio is not None:
            return DATA_SUBTYPES.AUDIO
    except Exception:
        # Not a file or file doesn't exist
        return None


def get_number_subtype(string):
    """ Returns the subtype inferred from a number string, or False if its not a number"""
    pytype = cast_string_to_python_type(str(string))
    if isinstance(pytype, float):
        return DATA_SUBTYPES.FLOAT
    elif isinstance(pytype, int):
        return DATA_SUBTYPES.INT
    else:
        return None


class TypeDeductor(BaseModule):
    """
    The type deduction phase is responsible for inferring data types
    from cleaned data
    """

    def count_data_types_in_column(self, data):
        type_counts = Counter()
        subtype_counts = Counter()
        additional_info = {}

        def type_check_numeric(element):
            type_guess, subtype_guess = None, None
            subtype = get_number_subtype(element)
            if subtype is not None:
                type_guess = DATA_TYPES.NUMERIC
                subtype_guess = subtype
            return type_guess, subtype_guess

        def type_check_date(element):
            type_guess, subtype_guess = None, None
            try:
                dt = parse_datetime(element)

                # Not accurate 100% for a single datetime str,
                # but should work in aggregate
                if dt.hour == 0 and dt.minute == 0 and \
                    dt.second == 0 and len(element) <= 16:
                    subtype_guess = DATA_SUBTYPES.DATE
                else:
                    subtype_guess = DATA_SUBTYPES.TIMESTAMP
                type_guess = DATA_TYPES.DATE
            except Exception:
                pass
            return type_guess, subtype_guess

        def type_check_sequence(element):
            type_guess, subtype_guess = None, None

            for sep_char in [',', '\t', '|', ' ']:
                all_nr = True
                if '[' in element:
                    ele_arr = element.rstrip(']').lstrip('[').split(sep_char)
                else:
                    ele_arr = element.rstrip(')').lstrip('(').split(sep_char)

                for ele in ele_arr:
                    if not get_number_subtype(ele):
                        all_nr = False
                        break

                if len(ele_arr) > 1 and all_nr:
                    additional_info['separator'] = sep_char
                    type_guess = DATA_TYPES.SEQUENTIAL
                    subtype_guess = DATA_SUBTYPES.ARRAY

            return type_guess, subtype_guess

        def type_check_file(element):
            type_guess, subtype_guess = None, None
            subtype = get_file_subtype_if_exists(element)
            if subtype:
                type_guess = DATA_TYPES.FILE_PATH
                subtype_guess = subtype
            return type_guess, subtype_guess

        type_checkers = [type_check_numeric,
                         type_check_date,
                         type_check_sequence,
                         type_check_file]
        for element in [str(x) for x in data]:
            for type_checker in type_checkers:
                data_type_guess, subtype_guess = type_checker(element)
                if data_type_guess:
                    break
            else:
                data_type_guess = 'Unknown'
                subtype_guess = 'Unknown'

            type_counts[data_type_guess] += 1
            subtype_counts[subtype_guess] += 1

        return type_counts, subtype_counts, additional_info

    def get_column_data_type(self, data, full_data, col_name):
        """
        Provided the column data, define its data type and data subtype.

        :param data: an iterable containing a sample of the data frame
        :param full_data: an iterable containing the whole column of a data frame

        :return: type and type distribution, we can later use type_distribution to determine data quality
        NOTE: type distribution is the count that this column has for belonging cells to each DATA_TYPE
        """
        additional_info = {'other_potential_subtypes': [], 'other_potential_types': []}

        if len(data) == 0:
            self.log.warning(f'Column {col_name} has no data in it. '
                             f'Please remove {col_name} from the training file or fill in some of the values !')
            return None, None, None, None, additional_info

        type_dist, subtype_dist = {}, {}

        # User-provided dtype
        if col_name in self.transaction.lmd['data_subtypes']:
            curr_data_type = self.transaction.lmd['data_types'][col_name]
            curr_data_subtype = self.transaction.lmd['data_subtypes'][col_name]
            type_dist[curr_data_type] = len(data)
            subtype_dist[curr_data_subtype] = len(data)
            self.log.info(f'Manually setting the types for column {col_name} to {curr_data_type}->{curr_data_subtype}')
            return curr_data_type, curr_data_subtype, type_dist, subtype_dist, additional_info

        # Forced categorical dtype
        if col_name in self.transaction.lmd['force_categorical_encoding']:
            curr_data_type = DATA_TYPES.CATEGORICAL
            curr_data_subtype = DATA_SUBTYPES.MULTIPLE
            type_dist[DATA_TYPES.CATEGORICAL] = len(data)
            subtype_dist[DATA_SUBTYPES.MULTIPLE] = len(data)
            return curr_data_type, curr_data_subtype, type_dist, subtype_dist, additional_info

        type_dist, subtype_dist, new_additional_info = self.count_data_types_in_column(data)

        if new_additional_info:
            additional_info.update(new_additional_info)

        # @TODO consider removing or flagging rows where data type is unknown in the future, might just be corrupt data...
        known_type_dist = {k: v for k, v in type_dist.items() if k != 'Unknown'}

        if known_type_dist:
            max_known_dtype, max_known_dtype_count = max(
                known_type_dist.items(),
                key=lambda kv: kv[0]
            )
        else:
            max_known_dtype, max_known_dtype_count = None, None

        nr_vals = len(full_data)
        nr_distinct_vals = len(set(full_data))

        # Data is mostly not unknown, go with type counting results
        if max_known_dtype and max_known_dtype_count > type_dist['Unknown']:
            curr_data_type = max_known_dtype

            possible_subtype_counts = [(k, v) for k, v in subtype_dist.items()
                                    if k in DATA_TYPES_SUBTYPES[curr_data_type]]
            curr_data_subtype, _ = max(
                possible_subtype_counts,
                key=lambda pair: pair[1]
            )
        else:
            curr_data_type, curr_data_subtype = None, None

        # Check for Tags subtype
        if curr_data_subtype != DATA_SUBTYPES.ARRAY:
            lengths = []
            unique_tokens = set()

            can_be_tags = False
            if all(isinstance(x, str) for x in data):
                can_be_tags = True
                delimiter = self.transaction.lmd.get('tags_delimiter', ',')
                for item in data:
                    item_tags = [t.strip() for t in item.split(delimiter)]
                    lengths.append(len(item_tags))
                    unique_tokens = unique_tokens.union(set(item_tags))

            # If more than 30% of the samples contain more than 1 category and there's more than 6 of them and they are shared between the various cells
            if can_be_tags and np.mean(lengths) > 1.3 and len(unique_tokens) >= 6 and len(unique_tokens)/np.mean(lengths) < (len(data)/4):
                curr_data_type = DATA_TYPES.CATEGORICAL
                curr_data_subtype = DATA_SUBTYPES.TAGS

        # Categorical based on unique values
        if curr_data_type != DATA_TYPES.DATE and curr_data_subtype != DATA_SUBTYPES.TAGS:
            if nr_distinct_vals < (nr_vals / 20) or nr_distinct_vals < 6:
                if (curr_data_type != DATA_TYPES.NUMERIC) or (nr_distinct_vals < 20):
                    if curr_data_type is not None:
                        additional_info['other_potential_types'].append(curr_data_type)
                        additional_info['other_potential_subtypes'].append(curr_data_subtype)
                    curr_data_type = DATA_TYPES.CATEGORICAL

        # If curr_data_type is still None, then it's text or category
        if curr_data_type is None:
            lang_dist = get_language_dist(data)

            # Normalize lang probabilities
            for lang in lang_dist:
                lang_dist[lang] /= len(data)

            # If most cells are unknown language then it's categorical
            if lang_dist['Unknown'] > 0.5:
                curr_data_type = DATA_TYPES.CATEGORICAL
            else:
                nr_words, word_dist, nr_words_dist = analyze_sentences(data)

                if 1 in nr_words_dist and nr_words_dist[1] == nr_words:
                    curr_data_type = DATA_TYPES.CATEGORICAL
                else:
                    curr_data_type = DATA_TYPES.TEXT

                    if len(word_dist) > 500 and nr_words / len(data) > 5:
                        curr_data_subtype = DATA_SUBTYPES.RICH
                    else:
                        curr_data_subtype = DATA_SUBTYPES.SHORT

                    type_dist = {curr_data_type: len(data)}
                    subtype_dist = {curr_data_subtype: len(data)}

                    return curr_data_type, curr_data_subtype, type_dist, subtype_dist, additional_info

        if curr_data_type == DATA_TYPES.CATEGORICAL and curr_data_subtype != DATA_SUBTYPES.TAGS:
            if nr_distinct_vals > 2:
                curr_data_subtype = DATA_SUBTYPES.MULTIPLE
            else:
                curr_data_subtype = DATA_SUBTYPES.SINGLE

        if curr_data_type in [DATA_TYPES.CATEGORICAL, DATA_TYPES.TEXT]:
            type_dist = {curr_data_type: len(data)}
            subtype_dist = {curr_data_subtype: len(data)}

        return curr_data_type, curr_data_subtype, type_dist, subtype_dist, additional_info

    def run(self, input_data):
        stats_v2 = defaultdict(dict)
        stats_v2['columns'] = set(input_data.data_frame.columns.values)
        stats_v2['columns'].update(self.transaction.lmd['columns_to_ignore'])
        stats_v2['columns'] = list(stats_v2['columns'])
        
        sample_settings = self.transaction.lmd['sample_settings']
        if sample_settings['sample_for_analysis']:
            sample_margin_of_error = sample_settings['sample_margin_of_error']
            sample_confidence_level = sample_settings['sample_confidence_level']
            sample_percentage = sample_settings['sample_percentage']
            sample_function = self.transaction.hmd['sample_function']

            sample_df = input_data.sample_df(sample_function,
                                             sample_margin_of_error,
                                             sample_confidence_level,
                                             sample_percentage)

            sample_size = len(sample_df)
            population_size = len(input_data.data_frame)
            self.transaction.log.info(f'Analyzing a sample of {sample_size} '
                                      f'from a total population of {population_size},'
                                      f' this is equivalent to {round(sample_size*100/population_size, 1)}% of your data.')
        else:
            sample_df = input_data.data_frame

        for col_name in sample_df.columns.values:
            col_data = sample_df[col_name].dropna()

            (data_type, data_subtype, data_type_dist,
             data_subtype_dist, additional_info) = self.get_column_data_type(col_data,
                                                                             input_data.data_frame[col_name],
                                                                             col_name)

            type_data = {
                'data_type': data_type,
                'data_subtype': data_subtype,
                'data_type_dist': data_type_dist,
                'data_subtype_dist': data_subtype_dist,
                'description': """A data type, in programming, is a classification that specifies which type of value a variable has and what type of mathematical, relational or logical operations can be applied to it without causing an error. A string, for example, is a data type that is used to classify text and an integer is a data type used to classify whole numbers."""
            }

            stats_v2[col_name]['typing'] = type_data
            stats_v2[col_name]['additional_info'] = additional_info

            # work with the full data
            stats_v2[col_name]['identifier'] = get_identifier_description(
                input_data.data_frame[col_name],
                col_name,
                data_type,
                data_subtype,
                additional_info['other_potential_subtypes']
            )

            if stats_v2[col_name]['identifier'] is not None:
                if col_name not in self.transaction.lmd['force_column_usage']:
                    if col_name not in self.transaction.lmd['predict_columns']:
                        if (self.transaction.lmd.get('tss', None) and
                                self.transaction.lmd['tss']['is_timeseries'] and
                                col_name in self.transaction.lmd['tss']['order_by']):
                            pass
                        else:
                            self.transaction.lmd['columns_to_ignore'].append(col_name)

            if data_subtype_dist:
                self.log.info(f'Data distribution for column "{col_name}" '
                              f'of type "{data_type}" '
                              f'and subtype "{data_subtype}"')
                try:
                    self.log.infoChart(data_subtype_dist,
                                       type='list',
                                       uid=f'Data Type Distribution for column "{col_name}"')
                except Exception:
                    # Functionality is specific to mindsdb logger
                    pass

        self.transaction.lmd['stats_v2'] = stats_v2
