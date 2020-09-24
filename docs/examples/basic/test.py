#
from mindsdb_native import *

mdb = Predictor(name='home_rentals')

#use the model to make predictions
result = mdb.predict(
    when_data={"number_of_rooms": 2, "sqft": 1100, 'location': 'great', 'days_on_market': 10, "number_of_bathrooms": 1})

print(result[0]['rental_price'])
print(result[0])

when = {"sqft": 700}
result = mdb.predict(
    when_data=when)
print(result[0]['rental_price'])
