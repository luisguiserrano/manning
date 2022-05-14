import numpy
from matplotlib import pyplot as plt

def plot_scatter(x_iterable, y_iterable, x_label = "", y_label = "",  legend = None, **kwargs):
    x_array = numpy.array(x_iterable)
    y_array = numpy.array(y_iterable)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if legend is not None:
        plt.legend(legend)
    plt.scatter(x_array, y_array, **kwargs)

def plot_decision_boundary_2D(matrix_features, array_labels, model, index_x = 0, index_y = 1):
    """
    index_x, index_y are the (x,y) column indices. Defaulted to the first two columns
    Assumes labels are multi-class, but not multi-label.
    """

    list_markers = ['s', '^', '.', ',', 'o', 'v',  '<', '>', '1', '2', '3', '4', '8', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_', 'P', 'X', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 'None', None, ' ', '']
    iter_markers = iter(list_markers) # allows us to call next()

    # Parameters
    plot_step = 0.01

    # define domain bounds
    x_min, x_max = matrix_features[:, index_x].min() - 1, matrix_features[:, index_x].max() + 1
    y_min, y_max = matrix_features[:, index_y].min() - 1, matrix_features[:, index_y].max() + 1

    # Create grid rows and lines using the defined scales
    plot_step = 0.1
    xx, yy = numpy.meshgrid(
        numpy.arange(x_min, x_max, plot_step), 
        numpy.arange(y_min, y_max, plot_step)
    )

    # concatenates arrays along second axis to form another grid
    Z = model.predict(numpy.c_[xx.ravel(), yy.ravel()])

    # reshape the predictions back into a grid
    zz = Z.reshape(xx.shape)

    # plot the grid of x, y and z values as a surface
    contour = plt.contourf(xx, yy, zz, cmap='RdBu')
    plt.colorbar(contour) # add a legend

    # create scatter plot for samples from each class
    list_classes = numpy.unique(array_labels)

    for _class in list_classes:
        # get row indexes for samples with this class
        row_ix = numpy.where(array_labels == _class)
        # create scatter of these samples
        plt.scatter(
            matrix_features[row_ix, index_x], 
            matrix_features[row_ix, index_y], 
            cmap="Paired", # plt.cm.RdYlBu,
            edgecolor="black",
            marker = next(iter_markers)
            # label=iris.target_names[i],
            )