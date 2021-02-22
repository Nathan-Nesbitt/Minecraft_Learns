from matplotlib import pyplot
from sklearn.tree import plot_tree


def plot_decision_tree(model, columns=None, location=None):
    """

    """
    pyplot.figure(figsize=(11, 6), dpi=150)
    plot_tree(model, feature_names=columns)

    pyplot.title("Decision Tree Splits", fontsize=14, color='#2E282A')
    pyplot.show()
    if location:
        pyplot.savefig(location)


def scatter(x, y, title, xlabel, ylabel, location=None):
    '''
    Scatter plot
    ---
    @param x: data vector
    @param y: data vector
    @param title: string title for graph
    @param xlabel, ylabel: string labels for x and y axis
    ---
    displays output plot
    '''
    pyplot.figure(figsize=(11, 6), dpi=150)
    pyplot.grid(color="#2E282A", alpha=0.5)
    pyplot.scatter(x, y, alpha=0.75, color="#F2CC8F")

    pyplot.title(title, fontsize=14, color='#2E282A')
    pyplot.xlabel(xlabel, fontsize=12, color='#2E282A')
    pyplot.ylabel(ylabel, fontsize=12, color='#2E282A')
    pyplot.xlim(0)

    pyplot.show()
    if location:
        pyplot.savefig(location)



def scatter_groups(x, y, groups, title, xlabel, ylabel, location=None):
    '''
    Scatter plot
    ---
    @param x: data vector with n observations
    @param y: data vector with n observations
    @param groups: data vector with n group assignmetns
    @param title: string title for graph
    @param xlabel, ylabel: string labels for x and y axis
    @param location: location to save the model
    ---
    displays (and saves if location given) output plot
    '''
    pyplot.figure(figsize=(11, 6), dpi=150)
    pyplot.grid(color="#2E282A", alpha=0.5)

    scatter = pyplot.scatter(x, y, c=groups, alpha=0.75, cmap="Set3")

    pyplot.title(title, fontsize=14, color='#2E282A')
    pyplot.xlabel(xlabel, fontsize=12, color='#2E282A')
    pyplot.ylabel(ylabel, fontsize=12, color='#2E282A')
    pyplot.xlim(0)

    pyplot.legend(*scatter.legend_elements(), title="Classes")
    
    pyplot.show()
    if location:
        pyplot.savefig(location)
