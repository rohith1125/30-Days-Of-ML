```py
import pandas as pd

melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
melbourne_data.columns
```
- **`read_csv`**: Reads a comma-separated values (CSV) file into a DataFrame.


- **`melbourne_data.columns`**:represents the columns of the dataframe.

## Few manipulations on the data
- **`melbourne_data.head()`**:represents the first 5 rows of the dataframe.
- **`melbourne_data.tail()`**:represents the last 5 rows of the dataframe.
- **`melbourne_data.info()`**:represents the information of the dataframe.
- **`melbourne_data.describe()`**:represents the statistical summary of the dataframe.
- **`melbourne_data.isnull().sum()`**:represents the number of null values in each column.
- **`melbourne_data.isnull().sum().sum()`**:represents the total number of null values in the dataframe.
- **`melbourne_data.shape`**:represents the number of rows and columns in the dataframe.
- **`melbourne_data.dtypes`**:represents the data types of each column.
- **`melbourne_data.columns`**:represents the columns of the dataframe.
- **`melbourne_data.columns[0]`**:represents the first column of the dataframe.
- **`melbourne_data.columns[-1]`**:represents the last column of the dataframe.
- **`melbourne_data.columns[1:3]`**:represents the first two columns of the dataframe.
- **`melbourne_data.columns[:-1]`**:represents all columns except the last column of the dataframe.
- **`melbourne_data.columns[1:]`**:represents all columns except the first column of the dataframe.
- **`melbourne_data.columns[:-2]`**:represents all columns except the first and last column of the dataframe.
- **`melbourne_data.columns[::-1]`**:represents all columns in reverse order.
- **`melbourne_data.columns[::2]`**:represents all columns in reverse order.
- **`melbourne_data.columns[::-2]`**:represents all columns in reverse order.


- ** `melbourne_data.dropna()`**:represents the dataframe with all rows with null values removed.

To select subset of dataframe, we can use the following methods:
1. Dot notation, which we use to select the "prediction target"
1. Selecting with a column list, which we use to select the "features"

## Selecting The Prediction Target
You can pull out a variable with dot-notation. This single column is stored in a Series, which is broadly like a DataFrame with only a single column of data.

We'll use the dot notation to select the column we want to predict, which is called the prediction target. By convention, the prediction target is called y. So the code we need to save the house prices in the Melbourne data is

``` py
y = melbourne_data.Price
```

**Basically, we're selecting the column with the name "Price" and storing it in the variable y.**


## Selecting The Features
The columns that are inputted into our model (and later used to make predictions) are called "features." In our case, those would be the columns used to determine the home price. Sometimes, you will use all columns except the target as features. Other times you'll be better off with fewer features.

For now, we'll build a model with only a few features. Later on you'll see how to iterate and compare models built with different features.

We select multiple features by providing a list of column names inside brackets. Each item in that list should be a string (with quotes).

Here is an example:

``` py
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt']
```
**Basically, we are selecting the columns with the names "Rooms", "Bathroom", "Landsize", "BuildingArea", and "YearBuilt" and storing them in the variable melbourne_features.**

By convention, this data is called X.

``` py
X = melbourne_data[melbourne_features]
```
**Bascially, we are selecting the columns with the names "Rooms", "Bathroom", "Landsize", "BuildingArea", and "YearBuilt" and storing them in the variable X.**

melbourne_data is the csv file we read in earlier.

Now we have a dataframe (X) with the features we want to use.

``` py
X.head()
```

Prints the first 5 rows of the dataframe.

``` py
X.tail()
```
Prints the last 5 rows of the dataframe.

``` py
X.describe()
```
Prints the statistical summary of the dataframe.As earlier, those stats about data like the mean, standard deviation etc.,

# Building a decision tree model
You will use the scikit-learn library to create your models. When coding, this library is written as sklearn, as you will see in the sample code. Scikit-learn is easily the most popular library for modeling the types of data typically stored in DataFrames.

The steps to building and using a model are:

- Define(): What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.
- Fit(): Capture patterns from provided data. This is the heart of modeling.
- Predict(): Just what it sounds like
- Evaluate(): Determine how accurate the model's predictions are.


Here is an example of defining a decision tree model with scikit-learn and fitting it with the features and target variable.

``` py
from sklearn.tree import DecisionTreeRegressor
# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model, fits the model to the training data
melbourne_model.fit(X, y)
```

Many machine learning models allow some randomness in model training. Specifying a number for random_state ensures you get the same results in each run. This is considered a good practice. You use any number, and model quality won't depend meaningfully on exactly what value you choose.

We now have a fitted model that we can use to make predictions.

In practice, you'll want to make predictions for new houses coming on the market rather than the houses we already have prices for. But we'll make predictions for the first few rows of the training data to see how the predict function works.

``` py
# Predict housing prices for the following new houses
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
# The predict function returns a numpy array. Convert to a list for ease of reading
# The predict function takes an array or a list of arrays.
# The predict function predicts the price for the given features
print(melbourne_model.predict(X.head()))
```