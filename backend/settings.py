from tensorflow import keras
import backend.util as util
from tensorflow.keras import activations


class Settings():
    """Data class containing all the settings for the visualization tool."""

    def __init__(self):
        self.layer = None
        self.learning_rate = 75.0
        self.iterations = 20
        self.blur = True
        self.decay = True
        self.rotate = True
        self.scale = 1
        self.blur_kernel_size = 2
        self.freq_penalization = True
        self.filter = 1
        self.groups = 6
        self.dict_path = None

    def init_model(self, model):
        """Initializes the model settings after the model was imported.

        Args:
            model: the imported model.
        """
        self.model = model
        self.input_width = model.inputs[0].shape[1]
        self.input_height = model.inputs[0].shape[2]
        self.conv_layers = []
        self.grp_layers = []

        for i in range(len(model.layers)):
            layer = model.layers[i]
            # Check for convolutional layer
            if isinstance(layer, keras.layers.Conv2D):
                self.conv_layers.append(Conv_Layer(
                    i, layer.name, layer.input[0].shape, layer.output.shape, int(layer.output.shape[3]), (int(layer.output.shape[1]), int(layer.output.shape[2]))))
            else:
                try:
                    if layer.activation == activations.relu:
                        self.grp_layers.append(Conv_Layer(i, layer.name, layer.input[0].shape, layer.output.shape, int(
                            layer.output.shape[3]), (int(layer.output.shape[1]), int(layer.output.shape[2]))))
                except AttributeError:
                    continue

    def print_layers(self):
        """Prints the information about the convolution layers."""
        for layer in self.conv_layers:
            print(layer)

    def update_input(self, path):
        """Updates the input for the model.

        Args:
            path: the filepath for the image to be imported.
        """
        self.input_img = keras.preprocessing.image.load_img(path, target_size=(
            self.input_width, self.input_height))
        self.input_data = util.prepare_input(self.input_img)

    def get_layer_by_name(self, name):
        """Finds the layer object specified by the given name.

        Args:
            name: the name of the desired layer.

        Returns:
            layer: conv_layer object of the specified layer.
        """
        for layer in self.conv_layers:
            if layer.name == name:
                return layer
        for layer in self.grp_layers:
            if layer.name == name:
                return layer

    def set_dict_path(self, path):
        """Updates the path for the feature visualization dictionary.

        Args:
            path: the new directory path.
        """
        self.dict_path = path

    def import_settings(self, path):
        """Updates the settings to match the imported settings.

        Args:
            path: the path to the settings to be imported.
        """
        f = open(path, "r")
        sett_str = f.read().split("|")
        self.learning_rate = float(sett_str[0])
        self.iterations = int(sett_str[1])
        self.scale = int(sett_str[2])
        self.blur_kernel_size = int(sett_str[3])
        self.blur = bool(sett_str[4])
        self.decay = bool(sett_str[5])
        self.rotate = bool(sett_str[6])
        self.freq_penalization = bool(sett_str[7])

    def export_settings(self, path):
        """Exports the settings.

        Args:
            path: the path the settings should be exported to.
        """
        f = open(path, "w")
        f.write(str(self.learning_rate) + "|")
        f.write(str(self.iterations) + "|")
        f.write(str(self.scale) + "|")
        f.write(str(self.blur_kernel_size) + "|")
        f.write(str(self.blur) + "|")
        f.write(str(self.decay) + "|")
        f.write(str(self.rotate) + "|")
        f.write(str(self.freq_penalization))
        f.close()


class Conv_Layer:
    """Data class representing a convolution layer with additional information to those provided by Keras layer type."""

    def __init__(self, index, name, in_shape, out_shape, filter_count, neuron_shape):
        self.index = index
        self.name = name
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.filter_count = filter_count
        self.neuron_shape = neuron_shape
        self.neuron_count = neuron_shape[0] * neuron_shape[1]

    def __str__(self):
        return "(" + str(self.index) + ", " + self.name + ", " + str(self.out_shape) + ")"

    def __repr__(self):
        return "(" + str(self.index) + ", " + self.name + ", " + str(self.out_shape) + ")"
