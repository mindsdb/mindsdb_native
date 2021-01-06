<h1 align="center">
	<img width="300" src="https://github.com/mindsdb/mindsdb_native/blob/stable/assets/MindsDBColorPurp@3x.png?raw=true" alt="MindsDB"> 
	<br>
</h1>

<p align="center">
   <a href="https://github.com/mindsdb/mindsdb_native/actions"><img src="https://github.com/mindsdb/mindsdb_native/workflows/MindsDB%20Native%20workflow/badge.svg" alt="MindsDB Native Workflow"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.6%20|%203.7|%203.8-brightgreen.svg" alt="Python supported"></a>
   <a href="https://pypi.org/project/mindsdb_native/"><img src="https://badge.fury.io/py/mindsdb-native.svg" alt="PyPi Version"></a>
  <a href="https://pypi.org/project/MindsDB/"><img src="https://img.shields.io/pypi/dm/mindsdb" alt="PyPi Downloads"></a>
  <a href="https://community.mindsdb.com/"><img src="https://img.shields.io/discourse/posts?server=https%3A%2F%2Fcommunity.mindsdb.com%2F" alt="MindsDB Community"></a>
  <a href="https://www.mindsdb.com/"><img src="https://img.shields.io/website?url=https%3A%2F%2Fwww.mindsdb.com%2F" alt="MindsDB Website"></a>
</p>

MindsDB is an Explainable AutoML framework for developers built on top of Pytorch. It enables you to build, train and test state of the art ML models in as simple as one line of code. [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Machine%20Learning%20in%20one%20line%20of%20code%21&url=https://www.mindsdb.com&via=mindsdb&hashtags=ai,ml,machine_learning,neural_networks)

<h2 align="center">
 <img width="600" src="https://github.com/mindsdb/mindsdb_native/blob/stable/assets/MindsDBTerminal.png?raw=true" alt="MindsDB">
</h2>


## Try it out

* [Installing MindsDB](https://docs.mindsdb.com/Installing/)
* [Learning from Examples](https://docs.mindsdb.com/tutorials/BasicExample/)
* [MindsDB Explainability GUI](https://docs.mindsdb.com/scout/Introduction/)
* [Frequently Asked Questions](https://docs.mindsdb.com/FAQ/)
* [Provide Feedback to Improve MindsDB](https://mindsdb.com/contact-us/)




### Installation


* **Desktop**: You can use MindsDB on your own computer in under a minute, if you already have a python environment setup, just run the following command:

```bash
 pip install mindsdb_native --user
```

>Note: Python 64 bit version is required. Depending on your environment, you might have to use `pip3` instead of `pip` in the above command.*

  If for some reason this fail, don't worry, simply follow the [complete installation instructions](https://docs.mindsdb.com/Installing/) which will lead you through a more thorough procedure which should fix most issues.

* **Docker**: If you would like to run it all in a container simply:  

```bash
sh -c "$(curl -sSL https://raw.githubusercontent.com/mindsdb/mindsdb/master/distributions/docker/build-docker.sh)"
```


### Usage

Once you have MindsDB installed, you can use it as follows:

Import **MindsDB**:

```python

from mindsdb_native import Predictor

```

One line of code to **train a model**:

```python
# tell mindsDB what we want to learn and from what data
Predictor(name='home_rentals_price').learn(
    to_predict='rental_price', # the column we want to learn to predict given all the data in the file
    from_data="https://s3.eu-west-2.amazonaws.com/mindsdb-example-data/home_rentals.csv" # the path to the file where we can learn from, (note: can be url)
)

```

One line of code to **use the model**:

```python

# use the model to make predictions
result = Predictor(name='home_rentals_price').predict(when_data={'number_of_rooms': 2, 'initial_price': 2000, 'number_of_bathrooms':1, 'sqft': 1190})

# you can now print the results
print('The predicted price is between ${price} with {conf} confidence'.format(price=result[0].explanation['rental_price']['confidence_interval'], conf=result[0].explanation['rental_price']['confidence']))

```

Visit the documentation to [learn more](https://docs.mindsdb.com/)

* **Google Colab**: You can also try MindsDB straight here [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg "MindsDB")](https://colab.research.google.com/drive/1qsIkMeAQFE-MOEANd1c6KMyT44OnycSb)


## Contributing

To contibute to MindsDB please checkout the [Contribution guide](https://github.com/mindsdb/mindsdb_native/blob/stable/CONTRIBUTING.md).

### Current contributors 

<a href="https://github.com/mindsdb/mindsdb_native/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=mindsdb/mindsdb_native" />
</a>

Made with [contributors-img](https://contributors-img.web.app).

## Report Issues

Please help us by [reporting any issues](https://github.com/mindsdb/mindsdb/issues/new/choose) you may have while using MindsDB.

## License

* [MindsDB License](https://github.com/mindsdb/mindsdb/blob/master/LICENSE)
