# Machine Learning Tools

## Data

Data is used to read in data from a file. The currently supported file types
are `.csv`, `.jsonl`, and `.json`.

It is initalized with a filepath
- `data = Data(filepath)`

The methods to save the data to a dataframe attribute are:
- `load_data()`: load the data at the determined filepath
- `load_csv(filepath)`: load a csv file at the input filepath
- `load_json(filepath)`: load a json file at the input filepath
- `load_json_lines(filepath)`: load a json lines file at the input filepath

It supports file deletion:
- `delete_file()`: delete the data file

After data has been loaded, the following methods are supported:
- `get_data(self)`: returns the pandas dataframe
- `print_data`: Prints the head of the data
- `print_types`: prints the datatypes of each column

## Common

### euclidean_distance
Computes the euclidean distance for every observation of data to an input observation

Inputs:
- `a`: numpy 2D array representing n observations of m predictors
- `b`: numpy array representing one observation with m predictors

Outputs:
- an array of n distances from a[i] to 

### interact
has a list of columns interact with each other, so `c = b * a`

Inputs:
- `data`: a dataframe of n observations and m predictors
- `interaction_cols`: list of columns to interact

Outputs:
- modified dataframe with added columns for the interactions

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

### mean_zero_normalize
Normalize the Data between -1 and 1 with mean 0

Inputs:
- `data`: a dataframe of n observations and m predictors

Outputs:
- a new dataframe with normalized values with range [-1,1] and mean 0

### normalize
Normalize the Data between 0 and 1

Inputs:
- `data`: a dataframe of n observations and m predictors

Outputs:
- a new dataframe with normalized values with range [0,1]

### standardize
standardize the data against its standard deviation with mean 0 and standard deviation 1

Inputs:
- `data`: a dataframe of n observations and m predictors

Outputs:
- a new dataframe with mean 0 and standard deviation 1

### label_encoding
encode the data at columns with string labels and return the label encoder

Inputs:
- `data`: a dataframe of n observations and m predictors

Outputs:
- `label_encoder`: encoder used for encoding and decoding
- `data`: dataframe with labels encoded

### encode_labels
encode the data using the label encoder

Inputs:
- `label_encoder`: encoder used for encoding and decoding
- `data`: a dataframe of n observations and m predictors

Outputs:
- a 2D array with encoded values

### log_transform
normalize and log transform the dataframe

Inputs:
- `data`: a dataframe of n observations and m predictors

Outputs:
- a new dataframe with log transfomed values

### is_dataframe
returns true of the input data is a dataframe

Inputs:
- `data`: a dataframe of n observations and m predictors

Outputs:
- boolean

### get_ith_column
get the ith column of a dataframe or numpy 2D array. If the input data is 1 dimenstional, returns the array

Inputs:
- `data`: a dataframe of n observations and m predictors

Ouptus:
- the column of values
- the name of the column, if data is 2 dimensional

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
- displays plot if no location is given
- saves the plot if a location is given

### plot_decision_tree
plots the decision tree used

Inputs:
- `model`: the *internal* model being used
- `columns`: a list of column names for features
- `location`: location to save the plot

Outputs:
- displays plot if no location is given
- saves the plot if a location is given

### scatter
creates a scatter plot

Inputs:
- `x`, `y`: data vectors to plot
- `title`: string title for graph
- `xlabel`, `ylabel`: string labels for x and y axes
- `location`: location to save the plot

Outputs:
- displays plot if no location is given
- saves the plot if a location is given

#### scatter_groups
creates a scatter plot for the groups found

Inputs:
- `x`, `y`: data vectors to plot
- `groups`: data vector with group assignments for each observation
- `title`: string title for graph
- `xlabel`, `ylabel`: string labels for x and y axes
- `location`: location to save the plot

Outputs:
- displays plot if no location is given
- saves the plot if a location is given

