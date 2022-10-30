import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow import keras
import tensorflow as tf
import os
from enum import Enum
import sys


def load_model(path):
    """Wrapper function for loading the keras model.
      Args:
        path: the path to the keras model.

    Returns:
        the imported keras model.
    """
    return keras.models.load_model(path, compile=False)


def deprocess_image(img):
    """Converts the given image numpy array with values between 0 and 1 to an rgb numpy array.
    Source: https://keras.io/examples/vision/visualizing_what_convnets_learn/.
    Args:
        img: the numpy array containing the image data to be converted.

    Returns:
        img: the numpy array in rgb format.
    """
    # Normalize array: center on 0., ensure variance is 0.15
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15

    # Center crop
    img = img[25:-25, 25:-25, :]

    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # Convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img


def prepare_input(img, size=None):
    """Preprocesses the input image for the keras model.
    Rescales the image to match the input size of the imported model and
    expands the image to 4 dimensions.

    If your model requires different preprocessing steps adjust it here.

    Args:
        img: the imported input image.

    Returns:
        img: the preprocessed image data.
    """
    if size is not None:
        img = tf.keras.preprocessing.image.smart_resize(
            img, size, interpolation='bilinear')
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    return x


def export_features(images, margin, n, name, settings):
    """Combines and saves the given array of images to a single image with the dimensions n * n.

    Args:
        images: list of images to be combined.
        margin: margin between the stitched images.
        n: dimension of the generated image.
        name: filename of the image to be stored.
        settings: visualization settings object.

    Returns:
        img: the preprocessed image data.
    """
    # Build a black picture with enough space for
    # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    if not os.path.exists("output"):
        os.makedirs("output")
    path = os.path.join("output", name)
    cropped_width = settings.input_width - 25 * 2
    cropped_height = settings.input_height - 25 * 2
    width = n * cropped_width + (n - 1) * margin
    height = n * cropped_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))
    # Fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            img = images[i * n + j]
            stitched_filters[
                (cropped_width + margin) * i: (cropped_width + margin) * i + cropped_width,
                (cropped_height + margin) * j: (cropped_height + margin) * j
                + cropped_height,
                :,
            ] = img
    keras.preprocessing.image.save_img(path, stitched_filters)


def combine_activation_grid(activation_grid, settings, worker, margin=0):
    """Combines the given array of images to a single image.

    Args:
        activation_grid: list of images to be combined.
        settings: visualization settings object.
        worker: the worker object that runs the task on a second thread. The object is used to emit progress to the main thread.
        margin: margin between the stitched images.

    Returns:
        stitched: the combined image.
    """
    n = len(activation_grid)
    cropped_width = settings.input_width - 25 * 2
    cropped_height = settings.input_height - 25 * 2
    width = n * cropped_width + (n - 1) * margin
    height = n * cropped_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))
    # Fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            if not worker.is_running:
                return None
            img = activation_grid[i][j]
            stitched_filters[
                (cropped_width + margin) * i: (cropped_width + margin) * i + cropped_width,
                (cropped_height + margin) * j: (cropped_height + margin) * j
                + cropped_height,
                :,
            ] = img
    return stitched_filters


def combine_group_img(settings, grid, vis):
    """Combines the given activation map and feature visualization to a single image.

    Args:
        settings: visualization settings object.
        grid: the activation map.
        vis: the feature visualization.

    Returns:
        stitched: the combined image.
    """
    stitched = np.zeros(
        (2*settings.input_width, 2*settings.input_height + vis.shape[1], 3))
    stitched[0:, 0:2*settings.input_height, :, ] = grid
    stitched[0:vis.shape[0], 2*settings.input_height: 2 *
             settings.input_height + vis.shape[1], :, ] = vis
    return stitched


def prepare_feature_extractor(model, layer_name):
    """Creates a modified model with an output of the activations for the given layer.

    Args:
        model: the original model.
        layer_name: the name of the layer whose activations shall be output.

    Returns:
        the modified model.
    """
    layer = model.get_layer(name=layer_name)
    inputs = model.inputs
    outputs = layer.output

    return keras.Model(inputs=inputs, outputs=outputs)


def get_layer_model(layer, settings):
    """Creates a modified model with an additional output of the activations for the given layer.

    Args:
        layer: the layer whose output shall be added.
        settings: visualization settings object.

    Returns:
        model: the modified model.
    """
    inputs = settings.model.inputs
    outputs = [settings.model.outputs, layer.output]
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_activations(model, layer, input_data):
    """Calculates the activations for the given input at the given layer.

    Args:
        model: the keras model.
        layer: the model the activations should be calculated for.
        input_data: the input for the model

    Returns:
        the activations of the given layer.
    """
    model = prepare_feature_extractor(model, layer)
    acts = model(input_data)
    return tf.squeeze(acts).numpy()


def import_img(path):
    """Imports the image from the given path.

    Args:
        path: the path of the image.

    Returns:
        img: PIL instance of the imported image.
    """
    try:
        img = image.load_img(path)
        return img
    except IOError as exc:
        raise


def convert_neuron_index(settings, neuron):
    """Converts 1D neuron index into 2D coordinates.

    Args:
        settings: visualization settings object.
        neuron: the neuron index to be converted

    Returns:
        tuple containing the x and y coordinates of the neuron.
    """
    neuron_shape = settings.get_layer_by_name(settings.layer).neuron_shape
    row = 0
    while neuron > neuron_shape[1]:
        neuron -= neuron_shape[1]
        row += 1
    return (row, neuron - 1)


def save_img(img, path):
    """Saves the given image at the given location.

    Args:
        img: the image to be save.
        path: the location the image shall be saved at.
    """
    img = deprocess_image(img[0].numpy())
    img = keras.preprocessing.image.array_to_img(img)
    img.save(path)


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class Target(str, Enum):
    """Enum for the differend loss targets for the feature visualization."""
    FILTER = "FILTER"
    NEURON = "NEURON"
    DIRECTION = "DIRECTION"

class Screen(str, Enum):
    """Enum for the differend screen types."""
    MAIN = "main"
    SAMPLE = "sample"
    FILTER_ACTS = "filter_acts"
    LAYER_REP = "layer_rep"
    GROUPED = "grouped"