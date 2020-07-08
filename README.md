<h1 align="center">
	<img width="300" src="https://github.com/mindsdb/mindsdb_native/blob/stable/assets/MindsDBColorPurp@3x.png?raw=true" alt="MindsDB"> 
	<br>
	
</h1>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.6%20|%203.7|%203.8-brightgreen.svg" alt="Python supported"></a>
   <a href="https://pypi.org/project/MindsDB/"><img src="https://badge.fury.io/py/MindsDB.svg" alt="PyPi Version"></a>
  <a href="https://pypi.org/project/MindsDB/"><img src="https://img.shields.io/pypi/dm/mindsdb" alt="PyPi Downloads"></a>
  <a href="https://community.mindsdb.com/"><img src="https://img.shields.io/discourse/posts?server=https%3A%2F%2Fcommunity.mindsdb.com%2F" alt="MindsDB Community"></a>
  <a href="https://www.mindsdb.com/"><img src="https://img.shields.io/website?url=https%3A%2F%2Fwww.mindsdb.com%2F" alt="MindsDB Website"></a>
</p>





MindsDB is an Explainable AutoML framework for developers built on top of Pytorch. It enables you to build, train and test state of the art ML models in as simple as one line of code. [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Machine%20Learning%20in%20one%20line%20of%20code%21&url=https://www.mindsdb.com&via=mindsdb&hashtags=ai,ml,machine_learning,neural_networks)

<table border="0" style=" border: 0px solid white;">
	<tbody border="0" style="border:0px">
<tr border="0" style="border:0px">
	<td border="0" style="border:0px">
		<img width="600" src="https://github.com/mindsdb/mindsdb_native/blob/stable/assets/MindsDBTerminal.png?raw=true" alt="MindsDB">
	</td>
	<td border="0" style="border:0px">
		<img alt="Linux build" src="https://www.screenconnect.com/Images/LogoLinux.png" align="center" height="30" width="30" />  <a href="https://travis-ci.com/mindsdb/mindsdb">
		<img src="https://badges.herokuapp.com/travis.com/mindsdb/mindsdb?branch=master&label=build&env=BADGE=linux"/>
		</a><hr/>
		<img alt="Windows build" src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/76/Windows_logo_-_2012_%28dark_blue%2C_lines_thinner%29.svg/414px-Windows_logo_-_2012_%28dark_blue%2C_lines_thinner%29.svg.png" align="center" height="30" width="30" /> <a href="https://travis-ci.com/mindsdb/mindsdb">
		<img src="https://badges.herokuapp.com/travis.com/mindsdb/mindsdb?branch=master&label=build&env=BADGE=windows"/>
		</a><hr/>
		<img alt="macOS build" src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/fa/Apple_logo_black.svg/245px-Apple_logo_black.svg.png"  align="center" height="30" width="30" /> <a href="https://travis-ci.com/mindsdb/mindsdb">
		<img src="https://badges.herokuapp.com/travis.com/mindsdb/mindsdb?branch=master&label=build&env=BADGE=osx"/>
		</a>
	</td>	
</tr>
	</tbody>
</table>




## Try it out

* [Installing MindsDB](https://docs.mindsdb.com/Installing/)
* [Learning from Examples](https://docs.mindsdb.com/tutorials/BasicExample/)
* [MindsDB Explainability GUI](http://mindsdb.com/product)
* [Frequently Asked Questions](https://docs.mindsdb.com/FAQ/)
* [Provide Feedback to Improve MindsDB](https://mindsdb.typeform.com/to/c3CEtj)




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


## Video Tutorial

Please click on the image below to load the tutorial:

[![here](https://img.youtube.com/vi/a49CvkoOdfY/0.jpg)](https://youtu.be/yr7fgqt9cfU)  

(Note: Please manually set it to 720p or greater to have the text appear clearly)

## MindsDB Graphical User Interface

You can also work with mindsdb via its graphical user interface ([download here](http://mindsdb.com/product)).
Please click on the image below to load the tutorial:

[![here](https://img.youtube.com/vi/fOwdv4j26CA/0.jpg)](https://youtu.be/fOwdv4j26CA)  


## MindsDB Lightwood: Machine Learning Lego Blocks

Under the hood of mindsdb there is lightwood, a Pytorch based framework that breaks down machine learning problems into smaller blocks that can be glued together seamlessly. More info about [MindsDB lightwood's on GITHUB](https://github.com/mindsdb/lightwood/).

## Contributing

In order to make changes to mindsdb, the ideal approach is to fork the repository than clone the fork locally `PYTHONPATH`.

For example: `export PYTHONPATH=$PYTHONPATH:/home/my_username/mindsdb`.

Too test your changes you can run unit tests (fast) and CI tests (slightly longer) locally.

To run unit tests:
* Install pytest: `pip install -r requirements_test.txt`
* Run: `pytest`

Once you have specific changes you want to merge into master, feel free to make a PR.

## Report Issues

Please help us by [reporting any issues](https://github.com/mindsdb/mindsdb/issues/new/choose) you may have while using MindsDB.

## License

* [MindsDB License](https://github.com/mindsdb/mindsdb/blob/master/LICENSE)
