import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.cm as cm

import random

"""Slightly adjusted version of the Grad-CAM implementation by Keras.
Source: https://keras.io/examples/vision/grad_cam/.
"""


def generate_gradcam(model, img, input_data, last_conv_layer):
    heatmap = make_gradcam_heatmap(model, input_data, last_conv_layer)
    return apply_heatmap(heatmap, img)


def make_gradcam_heatmap(model, img_array, layer, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(
            layer).output, model.output]
    )

    # Remove last layer's softmax
    grad_model.layers[-1].activation = None

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    layer_output = layer_output[0]
    heatmap = layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def apply_heatmap(heatmap, img, alpha=0.4):
    img = keras.preprocessing.image.img_to_array(img)
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(
        superimposed_img)

    # x = random.randint(0,100)
    # tf.keras.utils.save_img(f"gc{x}.png", superimposed_img)

    return superimposed_img

    # Save the superimposed image

    # x = random.randint(0,100)
    # tf.keras.utils.save_img(f"gc{x}.png", superimposed_img)
    # Display Grad CAM
    # display(Image(cam_path))
