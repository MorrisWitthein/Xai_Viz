import numpy as np

from backend.dictionary import Dictionary
import backend.activation_grid as ag
import backend.feature_visualization as fv
from backend.settings import Settings
import backend.grouper as grouper
import backend.util as util
from backend.util import Target
import backend.grad_cam as grad_cam


class Visualizer:
    """Interface managing communication between the front-end and the back-end of the tool."""
    def __init__(self):
        self.settings = Settings()
        self.dictionary = Dictionary(Target.FILTER)
        self.activations = None

    def update_input(self, path):
        """Updates the model's input image.
        
        Args:
            path: the path to the new input image.
        """
        self.settings.update_input(path)

    def visualize_filter(self):
        """Generates a feature visualization for a specified.
        
        Returns:
            the generated feature visualizations.
        """
        feature_extractor = util.prepare_feature_extractor(
            self.settings.model, self.settings.layer)
        return fv.visualize_filter(feature_extractor, self.settings, self.settings.filter)

    def visualize_neuron(self, neuron):
        """Generates a feature visualization for a specified.
        
        Args:
            neuron: the index of the neuron to be visualized.

        Returns:
            the generated feature visualizations.
        """
        feature_extractor = util.prepare_feature_extractor(
            self.settings.model, self.settings.layer)
        neuron_index = util.convert_neuron_index(self.settings, neuron)
        return fv.visualize_neuron(feature_extractor, self.settings.filter, neuron_index, self.settings)

    def generate_dictionary(self, layer, worker):
        """Generates a dictionary containing feature visualizations for the given layer.
        
        Args:
            layer: the current layer the generate visualizations for.
            worker: the worker object that runs the task on a second thread. The object is used to emit progress to the main thread. 
        """
        self.dictionary.generate_dictionary(self.settings, layer, worker)

    def reset_dictionary(self):
        """Clears the feature visualization dictionary."""
        self.dictionary = Dictionary(Target.FILTER)

    def update_activations(self):
        """Calculates the activations of the currently selected layer."""
        self.activations = util.get_activations(
            self.settings.model, self.settings.layer, self.settings.input_data)

    def update_dictionary(self, layer):
        """Imports the dictionary for the given layer.
        
        Args:
            layer: the layer to import the dictionary for.
        """
        self.dictionary.import_dictionary(self.settings.dict_path, layer)

    def update_settings(self, path):
        """Imports the settings from the given path.
        
        Args:
            path: the path to the settings.
        """
        self.settings.import_settings(path)

    def export_settings(self, path):
        """Exports the settings to the given path.
        
        Args:
            path: the path to export the settings to.
        """
        self.settings.export_settings(path)

    def get_filter_visualization(self, i, j, count=3):
        """Gets the feature visualizations for the top activated filter at the given spatial position.
        
        Args:
            i: row index of the considered positions.
            j: col index of the considered positions.
            count: specifies the number of filters to be returned.

        Returns:
            filter_visualizations: list of the top "count" activated filters at the specified position.
        """
        filter_visualizations = []
        max_idx = np.argsort(self.activations[j][i])
        imgs = self.dictionary.dictionary[self.settings.layer]
        for i in range(max_idx.shape[0] - 1, max_idx.shape[0] - count - 1, -1):
            filter_visualizations.append(imgs[max_idx[i]])
        return filter_visualizations

    def get_filter_visualizations(self, i, j, count=3):
        """Gets the feature visualizations for the top activated filter at the given spatial position.
        
        Args:
            i: row index of the considered positions.
            j: col index of the considered positions.
            count: specifies the number of filters to be returned.

        Returns:
            filter_visualizations: list of the top "count" activated filters at the specified position.
            filter_index: the indices of the filters.
        """
        filter_visualizations = []
        filter_index = []
        max_idx = np.argsort(self.activations[j][i])
        imgs = self.dictionary.dictionary[self.settings.layer]
        for i in range(max_idx.shape[0] - 1, max_idx.shape[0] - count - 1, -1):
            filter_visualizations.append(imgs[max_idx[i]])
            filter_index.append(max_idx[i])
        return (filter_visualizations, filter_index)

    def get_activation_grid(self, target, worker):
        """Generates an activation grid resembling a representation of the currently selected layer.
        
        Args:
            target: the target for the loss function for the feature visualization.
            worker: the worker object that runs the task on a second thread. The object is used to emit progress to the main thread. 

        Returns:
            The activation grid.   
        """
        activation_grid = []
        if target == Target.DIRECTION:
            activation_grid = ag.generate_activation_grid(
                self.settings, self.activations, worker)
        else:
            activation_grid = ag.generate_filter_activation_grid(
                self.activations, self.settings.layer, self.dictionary.dictionary, worker)
        if activation_grid == []:
            return None
        else:
            return util.combine_activation_grid(activation_grid, self.settings, worker)

    def apply_grad_cam(self, img=None, layer=None):
        """Applies Grad-CAM to the given image or layer representation.
        
        Args:
            img: the image to apply Grad-CAM to (input image if None is given).
            layer: the layer to calculated Grad-CAM for (input layer of None is given).

        Returns:
            the given image with Grad-CAM applied.
        """
        if layer is None:
            return grad_cam.generate_gradcam(self.settings.model, self.settings.input_img, self.settings.input_data, self.settings.conv_layers[-1].name)
        else:
            return grad_cam.generate_gradcam(self.settings.model, img, self.settings.input_data, layer)

    def generate_groups(self):
        """Generates activation groups for the current layer.
        
        Returns:
            the generated groups.
        """
        return grouper.generate_groups(self.settings)

    def generate_grp_visualizations(self, channel_factors):
        """Generates feature visualizations for the given activation groups.
        
        Args:
            channel_factors: the activations groups to generate visualizations for.

        Returns:
            list containing a feature visualization for each group.
        """
        return grouper.generate_grp_visualizations(channel_factors, self.settings.groups, self.settings)

    def generate_group_activation_maps(self, grouped_acts, worker):
        """Generates activation maps for the given groups.

        Args:
            grouped_acts: groups of activations to generate activation maps for.
            worker: the worker object that runs the task on a second thread. The object is used to emit progress to the main thread. 
        
        Returns:
            list containing an activation map for each group.
        """
        return grouper.generate_group_activation_maps(grouped_acts, self.settings, worker)
