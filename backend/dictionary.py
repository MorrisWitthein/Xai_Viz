from tensorflow.keras.preprocessing import image
from tensorflow import keras
import os
from tensorflow import keras
import json
import backend.feature_visualization as fv
from backend.util import Target
import backend.util as util


class Dictionary:
    """Dictionary containing the generated visualizations for a model."""

    def __init__(self, target):
        """
        Args:
            dictionary: dictionary mapping layer names to a list of visualizations for that layer
            target: states wether the visualizations resemble filters or neurons
        """
        self.dictionary = dict()
        self.target = target

    def generate_dictionary(self, settings, layer, worker):
        """Generates the feature visualizations for the given layer and appends them to the dictionary.
        Saves the images immediately to the disk to reduce memory usage.

        Args:
            layer: the layer the visualizations are generated for.
            settings: the current settings object.
            worker: the worker object that runs the task on a second thread. The object is used to emit progress to the main thread.
        """
        layer_name = layer.name
        feature_extractor = util.prepare_feature_extractor(
            settings.model, layer_name)
        if self.target == Target.FILTER:
            self.dictionary[layer_name] = self.generate_features(
                feature_extractor, settings, layer.filter_count, worker)
        elif self.target == Target.NEURON:
            self.dictionary[layer_name] = self.generate_features(
                feature_extractor, settings, layer.filter_count, layer.neuron_count, worker)
        self.export_dictionary(settings.dict_path, layer.name)
        self.dictionary.clear()

    def generate_features(self, feature_extractor, settings, filter_count, worker, neuron_count=None):
        """Depending on the target this function iterates over the filters (and neurons in the filter output if target==NEURON)
        and generates a visualization for each target.

        Args:
            feature_extractor: a modified model that outputs the activations for the selected layer.
            settings: the current settings object.
            filter_count: the number of filters in the selected layer.
            worker: the worker object that runs the task on a second thread. The object is used to emit progress to the main thread.
            neuron_count: the number of neurons in each filter output.

        Returns:
            all_imgs: the list of generated images
        """
        all_imgs = []
        for filter_index in range(filter_count):
            if worker.is_running and self.target == Target.FILTER:
                img = fv.visualize_filter(
                    feature_extractor, settings, filter_index)
                all_imgs.append(image.array_to_img(img))
            elif worker.is_running and self.target == Target.NEURON:
                neuron_imgs = []
                for i in range(neuron_count[0]):
                    for j in range(neuron_count[1]):
                        img = fv.visualize_neuron(
                            feature_extractor, filter_index, (i, j), settings)
                        neuron_imgs.append(image.array_to_img(img))
                all_imgs.append(neuron_imgs)
            elif not worker.is_running:
                return []
        return all_imgs

    def export_dictionary(self, path, layer):
        """Saves the generated dictionary of visualizations to the disk.
        Creates an index file in json format containing the selected target
        and the number of generated images for each filter.

        Args:
            path: the path the dictionary will be exported to.
            layer: the selected layer.
        """
        index = {
            "target": self.target,
            "layers": dict()
        }
        layers = index["layers"]

        if not os.path.exists(os.path.join(path, layer)):
            os.makedirs(os.path.join(path, layer))
        if self.target == Target.FILTER:
            layers[layer] = len(self.dictionary[layer])
            self.export_filter(path, layer)
        # elif self.target == Target.NEURON:
        #     layers[layer] = (len(self.dictionary[layer]),
        #                      len(self.dictionary[layer][0]))
        #     self.export_neurons(path, layer)
        if os.path.isfile(os.path.join(path, "index.json")):
            f = open(os.path.join(path, "index.json"))
            index = json.load(f)
            index["layers"][layer] = len(self.dictionary[layer])
        with open(os.path.join(path, "index.json"), "w") as index_file:
            json.dump(index, index_file)

    def export_filter(self, path, layer):
        """Writes the generated images to the disk.

        Args:
            path: the path the dictionary will be exported to.
            layer: the selected layer.
        """
        for filter_index, img in enumerate(self.dictionary[layer]):
            filter_path = os.path.join(path, layer,
                                       "filter_" + str(filter_index) + ".png")
            keras.preprocessing.image.save_img(
                filter_path, img)

    # def export_neurons(self, path, layer):
    #     for filter_index in range(len(self.dictionary[layer])):
    #         for neuron_index, img in enumerate(self.dictionary[layer][filter_index]):
    #             path = os.path.join(path, layer,
    #                                 "filter" + str(filter_index) + "_neuron" + str(neuron_index) + ".png")
    #             keras.preprocessing.image.save_img(path, img)

    def import_dictionary(self, import_path, layer):
        """Imports an already generated dictionary from the disk.
        To reduce memory usage, only one layer at a time is loaded.

        Args:
            path: the path the dictionary will be imported from.
            layer: the selected layer.

        """
        f = open(os.path.join(import_path, "index.json"))
        data = json.load(f)
        dictionary = dict()
        self.target = Target(data["target"])

        filter_count = data["layers"][layer]
        imgs = []
        if self.target == Target.FILTER:
            for i in range(filter_count):
                path = os.path.join(import_path, layer,
                                    "filter_" + str(i) + ".png")
                imgs.append(util.import_img(path))
        # elif self.target == Target.NEURON:
            #     for i in range(img_data[layer][0]):
            #         neuron_imgs = []
            #         for j in range(img_data[layer][1]):
            #             print(j)
            #             path = os.path.join(import_path, layer,
            #                                 "filter" + str(i) + "_neuron" + str(j) + ".png")
            #             neuron_imgs.append(self.import_img(path))
            #         imgs.append(neuron_imgs)
            dictionary[layer] = imgs
        self.dictionary.clear()
        self.dictionary = dictionary
