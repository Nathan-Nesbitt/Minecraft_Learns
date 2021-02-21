from matplotlib import pyplot
from sklearn.tree import plot_tree


def plot_decision_tree(model, columns=None, location=None):
    pyplot.figure(figsize=(11, 6), dpi=150)
    plot_tree(model, feature_names=columns)

    pyplot.title("Decision Tree Splits", fontsize=14, color='#2E282A')
    pyplot.show()
    if location:
        pyplot.savefig(location)
