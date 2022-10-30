import cv2
from tensorflow.keras.preprocessing import image
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import QObject, pyqtSignal
from backend.util import Target


class Worker(QObject):
    """Worker for running long task on a separate thread.
    Therefore the thread.started event gets connected with the task to be run.
    """
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def __init__(self, controller):
        super(QObject, self).__init__()
        self.controller = controller

    def run_generate_dictionary(self):
        """Generates a feature visualization for each filter in the network"""
        self.is_running = True
        self.completed = False
        i = 0
        # Generate visualizations for each Convolution Layer in the Network
        while self.is_running and i < len(self.controller.visualizer.settings.conv_layers):
            layer = self.controller.visualizer.settings.conv_layers[i]
            self.controller.visualizer.generate_dictionary(layer, self)
            i += 1
            self.progress.emit(i)
        # Check if the task was completed or canceled before emitting the finished signal
        if self.is_running:
            self.completed = True
            self.finished.emit()

    def run_generate_activation_grid(self):
        """Generates a layer representation for the selected Layer by choosing a filter visualization for each 
        pixel in the layer output and combining the to a single image.
        """
        self.is_running = True
        self.completed = False
        activation_grid = None
        layer_rep = self.controller.layer_rep_screen
        # Generate a layer representation depending on the selected loss target
        if layer_rep.target_choice.isChecked():
            activation_grid = self.controller.visualizer.get_activation_grid(
                Target.DIRECTION, self)
        else:
            activation_grid = self.controller.visualizer.get_activation_grid(
                Target.FILTER, self)
        # Exit here if the Process was canceled before completion
        if activation_grid is None:
            return
        layer_rep.img_array = cv2.resize(activation_grid, dsize=(
            layer_rep.img_size, layer_rep.img_size))
        # If the Grad-Cam option was enabled generate and apply heatmap to the generated Visualization
        if layer_rep.grad_cam.isChecked():
            layer_rep.update_input_image(
                self.controller.visualizer.apply_grad_cam())
            layer_rep.img_array = layer_rep.controller.visualizer.apply_grad_cam(
                layer_rep.img_array, layer_rep.controller.visualizer.settings.layer)
        else:
            layer_rep.update_input_image(
                self.controller.visualizer.settings.input_img)
        # Convert and display the generated visualization
        layer_rep.vis_img = image.array_to_img(layer_rep.img_array)
        layer_rep.vis_qim = ImageQt(layer_rep.vis_img)
        layer_rep.vis_pixmap = QtGui.QPixmap.fromImage(layer_rep.vis_qim)
        layer_rep.vis_container.setPixmap(layer_rep.vis_pixmap)
        # Check if the task was completed or canceled before emitting the finished signal
        if self.is_running:
            self.completed = True
            layer_rep.generated = True
            self.finished.emit()

    def run_generate_group_visualization(self):
        """Groups the activations of the selected layer and generates an activation map and a feature visualization for each group."""
        self.is_running = True
        self.completed = False
        group = self.controller.grouped_screen
        group.controller.visualizer.settings.groups = group.group_slider.value() * 2
        # Remove highlightings from the last visualization
        for i in range(len(group.group_vis_container)):
            group.group_vis_container[i].clear()
            group.group_vis_container[i].setStyleSheet("border: ")
        group.vis_container.clear()
        # Generate the activation groups
        groups = group.controller.visualizer.generate_groups()
        # Generate the activation map for the activation groups
        group.group_activation_maps = self.controller.visualizer.generate_group_activation_maps(
            groups[0], self)
        # If the task was canceled return here
        if group.group_activation_maps == []:
            return
        # Convert the activation maps to images and display the map for the first group
        for i, act_map in enumerate(group.group_activation_maps):
            act_map = cv2.resize(act_map, dsize=(
                group.img_size, group.img_size))
            group.group_activation_maps[i] = image.array_to_img(act_map)
        group.vis_qim = ImageQt(group.group_activation_maps[0])
        group.vis_pixmap = QtGui.QPixmap.fromImage(group.vis_qim)
        group.vis_container.setPixmap(group.vis_pixmap)
        # Generate and convert the Feature visualizations to images and display them
        group.grp_imgs = group.controller.visualizer.generate_grp_visualizations(
            groups[1])

        for i in range(len(group.grp_imgs)):
            img = Image.fromarray(group.grp_imgs[i])
            qim = ImageQt(img)
            pixmap = QtGui.QPixmap.fromImage(qim)
            group.group_vis_container[i].setPixmap(pixmap.scaled(
                100, 100, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))
        # Add the selected highlight to the first group
        group.selected_group = 0
        group.group_vis_container[group.selected_group].setStyleSheet(
            "border: 2px solid orange")
        # Check if the task was completed or canceled before emitting the finished signal
        if self.is_running:
            self.completed = True
            group.generated = True
            self.finished.emit()

    def stop(self):
        """Sets the running variable to cancel the running task."""
        self.is_running = False
