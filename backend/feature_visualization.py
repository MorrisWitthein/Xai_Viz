import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import cv2

import backend.util as util

"""Some regularizations inspired form: https://github.com/tensorflow/lucid.
Feature Visualization engine based on keras tutorial: https://keras.io/examples/vision/visualizing_what_convnets_learn/.
"""

def rand_select(xs):
    """Randomly selects rotation angle out of the given list.
    Args:
        xs: list of angles.

    Returns:
        tensor containing the selected angle
    """
    xs_list = list(xs)
    rand_n = tf.random.uniform((), 0, len(xs_list), "int32")
    return tf.constant(xs_list)[rand_n]


def random_rotate(t, angles):
    """Randomly rotates the given tensor image representation.

    Args:
        t: tensor representation of the image to be rotated.
        angles: list of angles to choose from.

    Returns:

    """
    t = tf.convert_to_tensor(t, dtype=tf.float32)
    angle = rand_select(angles)
    angle = angle2rads(angle)
    return tfa.image.rotate(t, angle)


def angle2rads(angle):
    """Converts angle to radiant.
    
    Args:
        angle: the angle to be converted.
    
    Returns:
        rad: the radiant representation of the angle.
    """
    angle = tf.cast(angle, "float32")
    rad = 3.14 * angle / 180.
    return rad


def penalize(img, settings, B=128):
    """Applies total variation and l1 penalties to the given image.
    
    Args:
        img: the image to be penalized.
        settings: the feature visualization settings.
        B: constant to control the strength of the penalty.

    Returns:
        img: the penalized image.
    """
    penalty = 1 / (settings.input_height *
                   settings.input_width *
                   B)
    tv = 1 / (settings.input_height *
              settings.input_width * 0.02 * B)
    img += penalty * B
    img += tv * tf.image.total_variation(img)
    return img


def decay_regularization(img, decay=0.8):
    """Applies decay penalty to the given image.
    
    Args:
        img: the image to be penalized.
        decay: the decay factor.

    Returns:
        the penalized image.
    """
    return decay * img


def blur_regularization(img, settings):
    """Applies blur to the given image.

    Args:
        img: the image to be blurred.
        settings: the feature visualization settings.
    
    Returns:
        blurred: the blurred image.    
    """
    tmp = np.transpose(img[0, :], (1, 0, 2))
    blurred = np.float32(
        cv2.blur(tmp, (settings.blur_kernel_size, settings.blur_kernel_size)))
    blurred = np.float32([np.transpose(blurred, (1, 0, 2))])
    blurred = tf.convert_to_tensor(blurred)
    return blurred


def compute_loss(feature_extractor, input_image, loss_f):
    """Calculates the given loss for the given input image.
    
    Args:
        feature_extractor: modified model to retrieve activations of the selected layer.
        input_image: the input to be optimized in the feature visualization process.
        loss_f: the loss function to be applied.

    Returns:
        the result of the loss function for the activations of the selected layer.
    """
    activation = feature_extractor(input_image)
    return loss_f(activation)


def gradient_ascent_step(feature_extractor, img, settings, loss_f):
    """Performs on step of the gradient ascent.
    
    Args:
        feature_extractor: modified model to retrieve activations of the selected layer.
        img: the input to be optimized in the feature visualization process.
        settings: the feature visualization settings.
        loss_f: the loss function to be applied.
    
    Returns:
        loss: the result of the given loss function.
        img: the img resulting after the gradient ascent step.
    """
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(feature_extractor, img, loss_f)
    # Compute gradients
    grads = tape.gradient(loss, img)

    # Normalize gradients
    grads = tf.math.l2_normalize(grads)
    img += settings.learning_rate * grads
    if settings.freq_penalization:
        img = penalize(img, settings)
    if settings.blur:
        img = blur_regularization(img, settings)
    if settings.decay:
        img = decay_regularization(img)
    if settings.rotate:
        img = random_rotate(img, list(range(-10, 11)) + 5 * [0])
    return loss, img


def initialize_image(settings):
    """Initializes a random image as a starting point for the feature visualization process.
    
    Args:
        settings: the feature visualization settings.

    Returns:
        the random image.
    """
    # Start from a gray image with some random noise
    img = tf.random.uniform(
        (settings.scale, settings.input_width, settings.input_height, 3))
    # Here we scale our random inputs to [-0.125, +0.125]
    return (img - 0.5 * settings.scale) * 0.25


def visualize_filter(feature_extractor, settings, filter_index):
    """Performs a feature visualization process for a filter target.
    
    Args:
        feature_extractor: modified model to retrieve activations of the selected layer.
        settings: the feature visualization settings.
        filter_index: the index of the target filter for the feature visualization process.

    Returns:
        img: the resulting image after the feature visualization process.
    """
    img = initialize_image(settings)
    loss_f = filter_loss(filter_index)
    for i in range(settings.iterations):
        loss, img = gradient_ascent_step(
            feature_extractor, img, settings, loss_f)

    # Decode the resulting input image
    img = util.deprocess_image(img[0].numpy())
    return img


def visualize_neuron(feature_extractor, filter_index, neuron_index, settings):
    """Performs a feature visualization process for a neuron target.
    
    Args:
        feature_extractor: modified model to retrieve activations of the selected layer.
        settings: the feature visualization settings.
        filter_index: the index of the target filter for the feature visualization process.
        neuron_index: the coordinates of the neuron in the given filter for the feature visualization process.

    Returns:
        img: the resulting image after the feature visualization process.
    """
    img = initialize_image(settings)
    loss_f = neuron_loss(filter_index, neuron_index)
    for _ in range(settings.iterations):
        _, img = gradient_ascent_step(feature_extractor, img, settings, loss_f)

    # Decode the resulting input image
    img = util.deprocess_image(img[0].numpy())
    return img


def visualize_direction(feature_extractor, acts, settings):
    """Performs a feature visualization process for the directions target.
    
    Args:
        feature_extractor: modified model to retrieve activations of the selected layer.
        acts: tensor containing the activations of all filters at a single spatial position.
        settings: the feature visualization settings.

    Returns:
        img: the resulting image after the feature visualization process.
    """
    img = initialize_image(settings)
    loss_f = direction_loss(acts)
    for _ in range(settings.iterations):
        _, img = gradient_ascent_step(feature_extractor, img, settings, loss_f)

    # Decode the resulting input image
    img = util.deprocess_image(img[0].numpy())
    return img


def filter_loss(filter_index):
    """Loss function for the filter target. Calculates the mean of the output tensor of the given filter.

    Args:
        filter_index: the index of the target filter.
        activation: the activations of the of the selected layer.

    Returns:
        inner: a function calculating the mean of the selected filter's activation.
    """
    def inner(activation):
        filter_activation = activation[:, 2:-2, 2:-2, filter_index]
        return tf.reduce_mean(filter_activation)
    return inner


def neuron_loss(filter_index, neuron_index):
    """Loss function for the neuron target. Calculates the activation of the given neuron.

    Args:
        filter_index: the index of the target filter.
        neuron_index: the index of the target neuron.
        activation: the activations of the of the selected layer.

    Returns:
        inner: a function calculating the selected filter's activation at the given neuron index.
    """
    def inner(activation):
        neuron_activation = activation[:,
                                       neuron_index[0], neuron_index[1], filter_index]
        return tf.reduce_sum(neuron_activation)
    return inner


def direction_loss(acts):
    """Loss function for the directions target.

    Args:
        acts: the activations for the input image
        activation: the activations for the generated visualization of the of the selected layer.

    Returns:
        inner: a function calculating the sum of the sum all filter's activations at a single spatial location scaled by their activation for the input image
    """
    def inner(activation):
        return tf.reduce_sum(activation * acts, -1)
    return inner
