import pytest
import requests
import mindsdb_native
import pandas as pd
import requests


@pytest.mark.integration
def test_quick_interface():
    predictor = mindsdb_native.Predictor(name='test_quick_interface')
    data_url = 'https://raw.githubusercontent.com/mindsdb/benchmarks/main/datasets/dow_jones/data.csv'
    df = pd.read_csv(data_url)
    df_train = df.iloc[:int(len(df)*0.8)]
    df_test = df.iloc[int(len(df)*0.8):]

    predictor.quick_learn(from_data=df_train, to_predict='next_weeks_close')

    predictions = predictor.quick_predict(df_test)
    assert len(predictions['next_weeks_close']) == len(df)
    for pred in predictions['next_weeks_close']:
        print(pred)
        assert isinstance(pred,float)
        assert pred > 0
