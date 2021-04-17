from sklearn.tree import plot_tree

from matplotlib import pyplot
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


BLACK = "#1F1F1F"
SINGLECLASSCOLOR = "#0E8543"

colors_continuous = ["#208120", "#3083F3"]
colors_groups = [
    "#8BF4E3", "#4CA1CC", "#3083F3", "#0E8543", "#208120", "#7ABB55",
    "#EAEA7A", "#E8BA5D", "#DE7C41", "#DE7C41", "#9F7323",
]

minecraft_linear = LinearSegmentedColormap.from_list(
    "minecraft_continuous", colors_continuous, N=64
)
minecraft_groups = ListedColormap(colors_groups, name="minecraft_groups")


def hist(population_list, title, xlabel, location=None):
    '''
    Create histogram for populations
    ---
    @param population_list: list of data vectors to graph
    @param title: string title for graph
    @param xlabel: string labels for x axis
    @param location: location to save the plot
    ---
    displays output plot
    '''
    pyplot.figure(figsize=(11,6), dpi=150)
    pyplot.grid(color=BLACK, alpha=0.5)

    # graph however many histograms are called
    if population_list.ndim == 1:
        pyplot.hist(
            population_list, bins=50, alpha=0.8, color=SINGLECLASSCOLOR
        )
    else:
        # plot every color
        for i in range(0, len(population_list.columns)):
            colors = ["red", "green", "blue"]
            pyplot.hist(
                population_list[i], bins=50, alpha=0.3, color=colors[i],
                label=population_list.columns[i]
            )
    pyplot.title(title, fontsize=14, color=BLACK)
    pyplot.xlabel(xlabel, fontsize=12, color=BLACK)
    pyplot.ylabel("Frequency", fontsize=12, color=BLACK)
    pyplot.legend()

    if location:
        pyplot.savefig(location)
    pyplot.show()


def plot_decision_tree(model, columns=None, location=None):
    """
    Plot the decision tree
    ---
    @param model: the *internal* model being used
    @param columns: a list of column names for features
    @param location: location to save graph
    """
    pyplot.figure(figsize=(11, 6), dpi=150)
    plot_tree(model, feature_names=columns)

    pyplot.title("Decision Tree Splits", fontsize=14, color=BLACK)

    if location:
        pyplot.savefig(location)
    pyplot.show()


def scatter(x, y, title, xlabel, ylabel, location=None):
    '''
    Scatter plot
    ---
    @param x: data vector
    @param y: data vector
    @param title: string title for graph
    @param xlabel, ylabel: string labels for x and y axis
    @param location: location to save graph
    ---
    displays output plot
    '''
    pyplot.figure(figsize=(11, 6), dpi=150)
    pyplot.grid(color=BLACK, alpha=0.5)

    diff_ybar = y - y.mean()
    pyplot.scatter(x, y, alpha=0.8, c=diff_ybar, cmap=minecraft_linear)

    pyplot.title(title, fontsize=14, color=BLACK)
    pyplot.xlabel(xlabel, fontsize=12, color=BLACK)
    pyplot.ylabel(ylabel, fontsize=12, color=BLACK)
    pyplot.xlim(0)
    pyplot.colorbar()

    if location:
        pyplot.savefig(location)
    pyplot.show()


def scatter_groups(x, y, groups, title, xlabel, ylabel, location=None):
    '''
    Scatter plot
    ---
    @param x: data vector with n observations
    @param y: data vector with n observations
    @param groups: data vector with n group assignmetns
    @param title: string title for graph
    @param xlabel, ylabel: string labels for x and y axis
    @param location: location to save graph
    ---
    displays (and saves if location given) output plot
    '''
    pyplot.figure(figsize=(11, 6), dpi=150)
    pyplot.grid(color=BLACK, alpha=0.5)

    scatter = pyplot.scatter(x, y, c=groups, alpha=0.8, cmap=minecraft_groups)

    pyplot.title(title, fontsize=14, color=BLACK)
    pyplot.xlabel(xlabel, fontsize=12, color=BLACK)
    pyplot.ylabel(ylabel, fontsize=12, color=BLACK)
    pyplot.xlim(0)

    pyplot.legend(*scatter.legend_elements(), title="Classes")
    
    if location:
        pyplot.savefig(location)
    pyplot.show()


def plot_accuracies(accuracies, hyperparams, location=None):
    """
    plot the accuracies over the hyperparams
    """
    pyplot.figure(figsize=(11, 6), dpi=150)
    pyplot.grid(color=BLACK, alpha=0.5)

    pyplot.plot(hyperparams, accuracies, color=SINGLECLASSCOLOR)

    pyplot.title('Comparision of Accuracies', fontsize=14, color=BLACK)
    pyplot.ylabel('Accuracy', fontsize=12, color=BLACK)
    pyplot.xlabel('Hyperparameter Value', fontsize=12, color=BLACK)

    if location:
        pyplot.savefig(location)
    pyplot.show()
