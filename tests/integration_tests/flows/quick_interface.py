import pytest
import requests
import mindsdb_native
import pandas as pd
import requests


@pytest.mark.integration
def test_quick_interface():
    predictor = mindsdb_native.Predictor(name='test_quick_interface')
    data_url = 'https://raw.githubusercontent.com/mindsdb/benchmarks/main/datasets/adult_income/adult.csv'
    df = pd.read_csv(data_url)
    df_train = df.iloc[:int(len(df)*0.8)]
    df_test = df.iloc[int(len(df)*0.8):]

    predictor.quick_learn(from_data=df_train, to_predict='hours-per-week', stop_training_in_x_seconds=20)

    predictions = predictor.quick_predict(df_test)
    print(predictions)
    assert len(predictions['hours-per-week']) == len(df)
    for pred in predictions['hours-per-week']:
        print(pred)
        assert isinstance(pred,int)
        assert pred > 0
