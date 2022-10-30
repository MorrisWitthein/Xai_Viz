from sklearn.decomposition import NMF
import numpy as np
from colorsys import hls_to_rgb
from PIL import Image
import tensorflow as tf

import backend.util as util
import backend.feature_visualization as fv

"""Inspired by: https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/building-blocks/NeuronGroups.ipynb"""


def generate_groups(settings):
    """Generates activation groups for the activations of the layer specified in the settings.

    Args:
        settings: the current settings object.

    Returns:
        grouped_acts: list containing the spatial activations of the extracted features.
        channel_factors: list containing the features extracted by the NMF.
    """
    model = NMF(n_components=settings.groups, init='random', random_state=0)
    layer_name = settings.layer
    input_data = settings.input_data
    feature_extractor = util.prepare_feature_extractor(
        settings.model, layer_name)
    acts = feature_extractor(input_data)
    grouped_acts = group_activations(
        acts, model).transpose(2, 0, 1).astype('float32')
    channel_factors = model.components_.astype('float32')
    x_peak = np.argmax(grouped_acts.max(1), 1)
    ns_sorted = np.argsort(x_peak)
    grouped_acts = grouped_acts[ns_sorted]
    channel_factors = channel_factors[ns_sorted]

    return (grouped_acts, channel_factors)


def group_activations(acts, model):
    """Calculates activation groups using sklearns NMF.

    Args:
        acts: the activations to be grouped.
        model: the examined model.

    Returns:
        the activation groups.        
    """
    acts = acts.numpy().squeeze()
    prev_shape = acts.shape
    acts_flat = acts.reshape([-1, acts.shape[-1]])
    new_flat = model.fit_transform(acts_flat)
    shape = list(prev_shape[:-1]) + [-1]
    return new_flat.reshape(shape)


def generate_grp_visualizations(channel_factors, n, settings):
    """Generates feature visualizations for the generated activation groups.

    Args:
        channel_factors: the activation groups.
        n = the number of groups.
        settings: the current settings object.

    returns:
        grp_imgs: list containing a feature visualization for each group.
    """
    feature_extractor = util.prepare_feature_extractor(
        settings.model, settings.layer)
    grp_imgs = []

    for i in range(n):
        img = fv.visualize_direction(
            feature_extractor, channel_factors[i], settings)
        grp_imgs.append(img)

    return grp_imgs


def normalize_array(array):
    """Normalizes the given array to consist of values between 0 and 100.

    Args:
        array: the array to be normalized.

    Returns:
        the normalized array.
    """
    low, high = np.min(array), np.max(array)
    domain = (low, high)

    if low < domain[0] or high > domain[1]:
        array = array.clip(*domain)

    min_value, max_value = np.iinfo(np.uint8).min, 100
    if np.issubdtype(array.dtype, np.inexact):
        offset = domain[0]
        if offset != 0:
            array -= offset
        if domain[0] != domain[1]:
            scalar = max_value / (domain[1] - domain[0])
        if scalar != 1:
            array *= scalar

        return array.clip(min_value, max_value).astype(np.uint8)


def generate_group_activation_maps(grouped_acts, settings, worker):
    """Generates activation maps for the given groups of activations.

    Args:
        grouped_acts: the activation groups the maps should be generated for.
        settings: the current settings object.
        worker: the worker object that runs the task on a second thread. The object is used to emit progress to the main thread.

    Returns:
        group_activations_maps: list containing an activation map for each of the provided groups.
    """
    group_activation_maps = []
    for k in range(settings.groups):
        acts = normalize_array(grouped_acts[k])
        activation_grid = []
        for x, row in enumerate(acts):
            activation_grid.append([])
            for _, val in enumerate(row):
                img = get_img(k, settings.groups, val)
                activation_grid[x].append(img)
        group_activation_maps.append(
            util.combine_activation_grid(activation_grid, settings, worker))
        if not worker.is_running:
            return []
        worker.progress.emit(k)
    return group_activation_maps


def get_img(i, n, act):
    """Converts the given activation into an image resembling an activation map.

    Args:
        i: the group index.
        n: the number of groups.
        act: the activation tensor.

    Returns:
        an image resembling the activation map.
    """
    hue = (i + 1 * 360) / n
    rgb = hls_to_rgb(hue, 0.5, act/100)
    rgb = [int(0.5 + 255*u) for u in rgb]
    return Image.new('RGB', (174, 174), tuple(rgb))


# def generate_group_attributions(grouped_acts, channel_factors, settings):
#     group_vecs = [grouped_acts[i, ..., None] * channel_factors[i]
#                   for i in range(settings.groups)]

#     attrs = np.asarray(class_group_attribution(
#         settings.input_data, settings.layer_name, 169, group_vecs))

#     return attrs

# def class_group_attribution(img, layer_name, label, group_vecs, settings):
#     grad_model = tf.keras.models.Model(
#         [settings.model.inputs], [settings.model.get_layer(
#             name=layer_name).output, settings.model.output]
#     )
#     with tf.GradientTape() as tape:
#         layer_output, preds = grad_model(img)
#         class_channel = preds[:, label]
#     grads = tape.gradient(class_channel, layer_output)
#     return [np.sum(group_vec * grads) for group_vec in group_vecs]
