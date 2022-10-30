import numpy as np
import tensorflow as tf
import backend.feature_visualization as fv
import backend.util as util


def generate_filter_activation_grid(activations, layer_name, dictionary, worker):
    """Creates a 2D-list of images as a representation for the given layer.
    Selects the filter with the highest activation for each spatial position of
    the activations of the given layer.

    Args:
        activations: the activations of the given layer.
        layer_name: the name of the layer.
        dictionary: dict containing the feature visualizations for the filter in the given layer.
        worker: the worker object that runs the task on a second thread. The object is used to emit progress to the main thread.

    Returns:
        activation_grid: a 2D-list of feature visualization with one image for each spatial position in the output of the given layer.
    """
    activation_grid = []
    imgs = dictionary[layer_name]
    for x, row in enumerate(activations):
        activation_grid.append([])
        for y, col in enumerate(row):
            if not worker.is_running:
                return []
            activation_grid[x].append(imgs[np.argmax(col)])
            worker.progress.emit(x*len(activations) + y)
    return activation_grid


def generate_activation_grid(settings, activations, worker):
    """Creates a list of images as a representation for the given layer.
    Generates an image for each spatial location in the output of the selected layer using the direction loss.

    Args:
        settings: the current settings object.
        activations: the activations of the given layer.
        worker: the worker object that runs the task on a second thread. The object is used to emit progress to the main thread.

    Returns:
        activation_grid: a list of feature visualization with one image for each spatial position in the output of the given layer.
    """
    feature_extractor = util.prepare_feature_extractor(
        settings.model, settings.layer)
    acts_flat = tf.squeeze(activations).numpy()
    acts_flat = acts_flat.reshape([-1] + [acts_flat.shape[2]])

    activation_grid = []
    row = []

    for n, v in enumerate(acts_flat):
        if not worker.is_running:
            return []
        if n > 0 and n % activations.shape[1] == 0:
            activation_grid.append(row)
            row = []
        img = fv.visualize_direction(feature_extractor, v, settings)
        row.append(img)
        worker.progress.emit(n)

    activation_grid.append(row)
    return activation_grid
