import random

from mindsdb_native.external_libs.stats import calculate_sample_size


def sample_data(df, sample_margin_of_error, sample_confidence_level, sample_percentage=None):
    population_size = len(df)
    if population_size <= 50:
        sample_size = population_size
    elif sample_percentage:
        assert sample_percentage > 0 and sample_percentage <= 100
        sample_size = int(round(len(df) * sample_percentage/100))
    else:
        sample_size = int(round(calculate_sample_size(population_size,
                                                  sample_margin_of_error,
                                                  sample_confidence_level)))

    population_size = len(df)
    input_data_sample_indexes = random.sample(range(population_size), sample_size)
    return df.iloc[input_data_sample_indexes]
