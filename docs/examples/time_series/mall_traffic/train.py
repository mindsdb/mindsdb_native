from sklearn.metrics import mean_squared_error
from mindsdb import Predictor
from helper import plotter
import pandas as pd


if __name__ == '__main__':
    train_data = pd.read_csv("mall_traffic_train.csv", index_col=False)
    test_data = pd.read_csv("mall_traffic_predict.csv", index_col=False)
    target = 'people_count'

    p = Predictor(name='mall_traffic')

    p.learn(
        from_data=train_data,
        to_predict=target,
        timeseries_settings={
            'order_by': ['TimeStamp'],
            'window': 6,  # consider last hour worth of measurements
            'use_previous_target': True
            }
        )

    forecast = p.predict(when_data=test_data)

    mse = mean_squared_error(forecast._data[f'__observed_{target}'], forecast._data[f'{target}'])

    print(f"\n\n[ Mall traffic ]\n\tRMSE: {round(mse**(1/2), 1)}\n\n")

    plotter(test_data['TimeStamp'], 
            forecast._data[f'__observed_{target}'], 
            forecast._data[f'{target}'])
