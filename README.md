# XAI Viz

XAI Viz is a visualization tool for convolutional Keras models. It uses [Feature Visualization](https://distill.pub/2017/feature-visualization/) and [Grad-CAM](https://arxiv.org/abs/1610.02391) to create visual representations of hidden layers of convolutional neural networks.

# Features

- Generate Feature Visualizations for filters or individual neurons.
- Investigate filter activations at spatial locations on your input image.
- Generate visual representations of hidden layers using Feature Visualization.
- Show the attribution to the models prediction of spatial locations on these representations using Grad-CAM.
- Form groups of activations using non-negative matrix factorization.
- Visualize these groups using Feature Visualization and activation maps.

# Installation

- If your model requires special input preprocessing clone the repo and update the prepare_input function in backend.util to your needs.
- After that install the requirements.txt and run the install script to generate an executable or simply run main.py.

- This application is designed for and tested with python 3.7 and tensorflow 2.7.

# Getting Started

- Export your Keras model using tf.keras.models.save_model()
- Start the tool and import your model
- Generate/Import a dictionary containing Feature Visualizations for each filter in your model (generating may take some time depending on your models complexity)
- Load an input and start visualizing

# Credits

- This tool implements many of the ideas proposed by [Olah et al.](https://distill.pub/2018/building-blocks/)
- [Feature Visualization](https://keras.io/examples/vision/visualizing_what_convnets_learn/) and [Grad-CAM](https://keras.io/examples/vision/grad_cam/) engines are based on the tutorials provided by Keras.
- Color scheme: [qdarkstyle](https://github.com/ColinDuquesnoy/QDarkStyleSheet).
