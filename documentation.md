# Machine Learning Tools

## Data

Data is used to read in data from a file. The currently supported file types
are `.csv`, `.jsonl`, and `.json`.

It is initalized with a filepath
- `data = Data(filepath)`

The methods to save the data to a dataframe attribute are:
- `load_data()`: load the data at the determined filepath. Data must be saved as ".csv", ".json" or ".jsonl"
- `load_csv(filepath)`: load a csv file at the input filepath
- `load_json(filepath)`: load a json file at the input filepath
- `load_json_lines(filepath)`: load a json lines file at the input filepath

It supports file deletion:
- `delete_file()`: delete the data file

After data has been loaded, the following methods are supported:
- `get_data(self)`: returns the pandas dataframe
- `print_data()`: Prints the head of the data
- `print_types()`: prints the datatypes of each column

## Common

### euclidean_distance
Computes the euclidean distance for every observation of data to an input observation

Inputs:
- `a`: numpy array
- `b`: numpy array

Outputs:
- an array of n distances from a[i] to 

```python
a = numpy.array([[1,1,1], [2,2,2]])
b = numpy.array([[2,2,2], [1,1,1]])

dist = euclidean_distance(a, b)
```

### interact
has a list of columns interact with each other, so `c = b * a`

Inputs:
- `data`: a dataframe of n observations and m predictors
- `interaction_cols`: list of columns to interact

Outputs:
- modified dataframe with added columns for the interactions

```python
data = DataFrame([[1,2],[3,4]],columns=["a", "b"])
interaction_cols = ["a", "b"]

new_data = interact(data, interaction_cols)
```

### one_hot_encode
replace a list of columns with the one hot encoded columns

Inputs:
- `data`: a dataframe of n observations and m predictors
- `encode_cols`: list of columns to encode

Outputs:
- a new dataframe where the unencoded columns are replaced with the encoded

### pca
transform the data with pca

Inputs:
- `data`: a dataframe of n observations and m predictors
- `n_components`: the number of components to keep. If None, keep all features

Outputs:
- a new array of transformed values

```python
pca_data = pca(data, 2)
```

### mean_zero_normalize
Normalize the Data between -1 and 1 with mean 0

Inputs:
- `data`: a dataframe of n observations and m predictors

Outputs:
- a new dataframe with normalized values with range [-1,1] and mean 0

```python
normalized_data = mean_zero_normalize(data)
```

### normalize
Normalize the Data between 0 and 1

Inputs:
- `data`: a dataframe of n observations and m predictors

Outputs:
- a new dataframe with normalized values with range [0,1]

```python
normalized_data = normalize(data)
```

### standardize
standardize the data against its standard deviation with mean 0 and standard deviation 1

Inputs:
- `data`: a dataframe of n observations and m predictors

Outputs:
- a new dataframe with mean 0 and standard deviation 1

```python
normalized_data = standardize(data)
```

### label_encoding
encode the data at columns with *string* labels and return the label encoder

Inputs:
- `data`: a dataframe of n observations and m predictors

Outputs:
- `label_encoder`: encoder used for encoding and decoding
- `data`: dataframe with labels encoded

```python
label_encoder, data = label_encoding(data)
```

### encode_labels
encode the data using the label encoder

Inputs:
- `label_encoder`: encoder used for encoding and decoding
- `data`: a dataframe of n observations and m predictors

Outputs:
- a 2D array with encoded values

```python
encoded_data = encode_labels(label_encoder, data)
```

### log_transform
normalize and log transform the dataframe

Inputs:
- `data`: a dataframe of n observations and m predictors

Outputs:
- a new dataframe with log transfomed values

```python
log_data = log_transform(data)
```

### is_dataframe
returns true of the input data is a dataframe

Inputs:
- `data`: a dataframe of n observations and m predictors

Outputs:
- boolean

```python
data = DataFrame()
true_df = is_dataframe(data)

false_df = is_dataframe(data.values)
```

### get_ith_column
get the ith column of a dataframe or numpy 2D array. If the input data is 1 dimenstional, returns the array

Inputs:
- `data`: a dataframe of n observations and m predictors

Ouptus:
- the column of values
- the name of the column, if data is 2 dimensional

```python
column_vals, column_name = get_ith_column(data)
```

## Graphing

### Color Maps
The following colors are available for use:
- `minecraft_linear`: a linear gradient of colours from grass green to sky blue
- `minecraft_groups`: a list mapping for grouping by color

### hist
create a histogram for a list of populations

Inputs:
- `population_list`: list of data vectors to graph
- `title`: string title for graph
- `xlabel`: string labels for x axis
- `location`: location to save the plot

Outputs:
- displays plot
- saves the plot if a location is given

```python
hist(population_list, "title", "xlabel", "plot.png")
```

### plot_decision_tree
plots the decision tree used

Inputs:
- `model`: the *internal* model being used
- `columns`: a list of column names for features
- `location`: location to save the plot

Outputs:
- displays plot
- saves the plot if a location is given

```python
plot_decision_tree(tree, [feature_1, feature_2], "plot.png")
```

### scatter
creates a scatter plot

Inputs:
- `x`, `y`: data vectors to plot
- `title`: string title for graph
- `xlabel`, `ylabel`: string labels for x and y axes
- `location`: location to save the plot

Outputs:
- displays plot
- saves the plot if a location is given

```python
scatter(x, y, "title", "xlabel", "ylabel", "plot.png")
```

#### scatter_groups
creates a scatter plot for the groups found

Inputs:
- `x`, `y`: data vectors to plot
- `groups`: data vector with group assignments for each observation
- `title`: string title for graph
- `xlabel`, `ylabel`: string labels for x and y axes
- `location`: location to save the plot

Outputs:
- displays plot
- saves the plot if a location is given

```python
scatter(x, y, labels, "title", "xlabel", "ylabel", "plot.png")
```

## Machine Learning Models

Minecraft Learns has several classification and regression models/

Classification Models:
- Decision Tree Classification: `DecisionTreeClassifier`
- KMeans Classification: `KMeans`
- K-Nearest Neighbors Classification: `KNN`
- Linear Discriment Analysis: `LDA`
- Random Forest Regression: `RandomForestClassifier`
- Support Vector Machine Classification: `SVMClassification`

Regression Models:
- Decision Tree Regression: `DecisionTreeRegression`
- Linear Regression: `LinearRegression`
- Partial Least Squares Regression: `PLSRegressor`
- Random Forest Regression: `RandomForestRegressor`
- Support Vector Machine Regression: `SVMRegression`

### ML Models:

Here is an example of how to use a machine learning model:
```python
tree = DecisionTreeRegression()
tree.process_data(X_train, y_train)
tree.train()

predictions = tree.predict(test_X)
cross_val = tree.evaluate()
```

All Machine Learning Models have the following functions:

#### hyperparameters
Each model has hyperparameters set on initalization.

- `internal_model`: sets the internal sklearn model object
- `pca`: tells whether to run pca on the model
- `score`: the cross validation accuracy of the model

Each is set to the standard setting for the model

#### set_parameters

This is an extension of the sklearn `set_params` method.

Inputs:
- a dictionary of hyperparameters and their appropriate new settings

Results:
- sets the parameters of the internal model using sklearn
- sets object parameters to input parameters in dictionary

```python
model.set_parameters({"k":10})
```

#### set_X and set_y
Set the internal training data to an input dataset. This does not do any preprocessing to this data

Input:
- `data`: data to save

```python
X = DataFrame([[1,0,1],[2,3,1]], columns=["a","b","c"])
model.set_X(X)
```

#### process_data
Does standard data preprocessing for the model and then saves resulting training data

Inputs:
- `X`: the training predictors
- `y`: the training responses

```python
model.process_data(X, y)
```

#### train
Trains the input data and evaluates the cross evaluation error on the model

```python
model.train()
```

#### predict
Predict response for input testing data

```python
predictions = model.predict(test_X)
```

#### evaluate
Return the cross validation score

```python
cross_val_score = model.evaluate()
```

#### save_model
Save the model to a `.sav` file at a given location

Inputs:
- `filename`: location to save model

```python
model.save_model("model.sav")
```

#### load_model
load model from a `.sav` file at a given location

Inputs:
- `filename`: location to save model

```python
model.load_model("model.sav")
```

### Classification Models

In addition, classification models have the following functions:

#### predict_probablity
Predict the probablity of label y for the input data X

Inputs:
- `X`: a 2D data matrix of n observations and m predictors

Outputs: 
- probably of label label

```python
probas = model.predict_probablity(test_X)
```

#### Labeling Metrics
The following labeling metrics are included:
- `misclassification_rate`: returns the misclassification rate of a test set
- `accuracy`: return accuracy score of the model on a test set
- `precision`: return the precision score of the model

Each takes the following inputs:
- `test_X`: dataframe of n test predictor observations
- `test_y`: dataframe of n test response observations

```python
accuracy_score = model.accuracy(test_X, test_y)
```

#### plot_precision_recall
plot the precesion recall curve

Inputs:
- `test_X`: dataframe of n test predictor observations
- `test_y`: dataframe of n test response observations
- `location`: *optional* if a location is set, the model is saved there

```python
model.plot_precision_recall(test_X, test_y, "model.png")
```

#### plot
make a scatter plot of the groups. See `scatter_groups`.

Inputs:
- `location`: *optional* if a location is set, the model is saved there

```python
model.plot("model.png")
```

### Regression Models

Each regression model has the following methods:

#### r_square
Calculate r square score as a measure of fir

Outputs:
- `r`: R^2 measure of fit

```python
r = model.r_square()
```

#### mse
evaluate the preformance of the model using MSE

Outputs:
- `mse`: mean squared error of the model

```python
mse = model.mse()
```

#### plot
make a scatter plot of the predictions. See `scatter`.

Inputs:
- `location`: *optional* if a location is set, the model is saved there

```python
model.plot("model.png")
```

### Decision Trees
Hyperparameters:
- `pca`: denotes of using PCA in preprocessing. Set to `False` by defaut

Both decision tree models have the following additional methods:

#### feature_importance
Returns the importance of each feature

```python
importance = tree.feature_importance()
```

#### plot
Plots either the decision tree itself (see `plot_tree`) or a scatter plot from the super plot method.

Inputs:
- `tree`: *optional* notes if plot_tree should be used. Set to `True` by default
- `location`: *optional* if a location is set, the model is saved there

```python
plot_tree = True
tree.plot(plot_tree, "model.png")
```

### KMeans:
Hyperparameters:
- `n_neighbors`: the number of neighbors to consider. Defaults to `3`.
- `pca`: denotes of using PCA in preprocessing. Set to `False` by defaut

### KNN
Hyperparameters:
- `n_clusters`: the number of clusters to create. Defaults to `3`.
- `pca`: denotes of using PCA in preprocessing. Set to `False` by defaut

### LDA
Hyperparameters:
- `pca`: denotes of using PCA in preprocessing. Set to `False` by defaut

The lda model inherits the following methods from the sklearn implementation of LDA:
- `mean`: return the overall mean of the model
- `class_means`: return the means of each class
- `get_intercept`: return the intercept of the model
- `get_coefficents`: return the coefficents of the model


### Linear Regression
Hyperparameters:
- `pca`: denotes of using PCA in preprocessing. Set to `False` by defaut
- `interactions`: denotes if having interacting features. Set to `None` by defaut
- `one_hot_encode`: denotes if one hot encoding features. Set to `None` by defaut

The linear regression inherits the following methods from the sklearn implementation of Linear Regression:
- `get_intercept`: return the intercept of the model
- `get_coefficents`: return the coefficents of the model

#### equation
returns a string of the equation used

```python
equation_string = lr.equation()
print(equation_string)
```

### Partial Least Squares Regression
Hyperparameters:
- `n_components`: number of components to keep in transformation. Set to `3` by default
- `one_hot_encode`: denotes if one hot encoding features. Set to `None` by defaut

The PLS regression inherits the following methods from the sklearn implementation of Linear Regression:
- `get_intercept`: return the intercept of the model
- `get_coefficents`: return the coefficents of the model

#### equation
returns a string of the equation used

```python
equation_string = pls.equation()
print(equation_string)
```

#### transform_data
Transform the data X into the new dimension found from PLS

Inputs:
- `X`: a 2D data matrix of n observations and m predictors in the origin space

Outputs:
- a 2D data matrix of n observations and m predictors in the transformed space

```python
transformed_data = pls.transform_data(data)
```

#### inverse_transform_data
Revert the data fro, the new dimension found from PLS into the original

Inputs:
- `X`: a 2D data matrix of n observations and m predictors in the transformed space

Outputs:
- a 2D data matrix of n observations and m predictors in the origin space

```python
data = pls.inverse_transform_data(transformed_data)
```

### Random Forest
Hyperparameters:
- `pca`: denotes of using PCA in preprocessing. Set to `False` by default

Both random forest models have the following additional method:

#### feature_importance
Returns the importance of each feature

```python
importance = tree.feature_importance()
```

### Support Vector Machines
Hyperparameters:
- `kernel`: kernel to use in SVM. Default is `rbf` for the gaussian kernel
- `degree`: degree of polynomial used for `poly` kernel. Ignored for other kernels. Defaults to `3`.
- `pca`: denotes of using PCA in preprocessing. Set to `False` by defaut
- `one_hot_encode`: denotes if one hot encoding features. Set to `None` by defaut

Both SVM models have the following additional methods:

#### get_support_vectors
Return the support vectors of the model

```python
support_vectors = svm.get_support_vectors()
```

#### get_n_support_vectors
Return the number of support vectors of the model

```python
n_sv = svm.get_n_support_vectors()
```

#### set_c
Set the C learning parameter

Inputs:
- `C`: value of c

```python
c = 10
svm.set_c(c)
```

#### set_best_c
Test the following values of C `[0.1, 1, 5, 10, 20, 50, 100]` and set the learning
parameter to the highest preforming value of c.

```python
svm.set_best_c()
```

