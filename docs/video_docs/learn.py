import mindsdb_native

model = mindsdb_native.Predictor(name='wine_model')
model.learn(from_data='wine_data_train.tsv', to_predict='Cultivar')
