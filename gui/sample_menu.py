from tensorflow import keras
from PyQt5 import QtWidgets, QtGui, uic, QtCore
import re
import os
from PIL import Image
from PIL.ImageQt import ImageQt

from backend.util import Screen
import backend.util as util


class Sample_Menu(QtWidgets.QWidget):
    """GUI class for the sample menu.
    The purpose of this screen is to optimize the settings for the feature visualization.
    Therefore sample images for filters across the network can be generated using the selected
    settings.
    """

    def __init__(self, controller):
        super(Sample_Menu, self).__init__()
        self.controller = controller
        self.img_array = None
        self.neuron_enabled = False
        self.neuron = 0
        path = util.resource_path(os.path.join(
            'res', 'ui', 'sample_screen.ui'))
        uic.loadUi(path, self)
        self.init_elements()

    def init_elements(self):
        """Initialize the UI elements and connect them with their corresponding functions."""
        self.vis_container = self.findChild(QtWidgets.QLabel, "vis_container")
        self.choose_layer = self.findChild(QtWidgets.QComboBox, "choose_layer")
        self.choose_layer.currentTextChanged.connect(self.update_layer)
        self.filter = self.findChild(QtWidgets.QSlider, "filter")
        self.selected_filter = self.findChild(
            QtWidgets.QLineEdit, "selected_filter")
        self.filter.valueChanged.connect(self.update_filter)
        self.selected_filter.textChanged.connect(self.update_filter)
        self.learning_rate = self.findChild(
            QtWidgets.QLineEdit, "learning_rate")
        self.learning_rate.textChanged.connect(self.update_learning_rate)
        self.iterations = self.findChild(QtWidgets.QLineEdit, "iterations")
        self.iterations.textChanged.connect(self.update_iterations)
        self.kernel = self.findChild(QtWidgets.QLineEdit, "kernel")
        self.kernel.textChanged.connect(self.update_kernel)
        self.blur = self.findChild(QtWidgets.QCheckBox, "blur")
        self.blur.stateChanged.connect(self.update_blur)
        self.freq_pen = self.findChild(QtWidgets.QCheckBox, "freq_pen")
        self.freq_pen.stateChanged.connect(self.update_freq_pen)
        self.decay = self.findChild(QtWidgets.QCheckBox, "decay")
        self.decay.stateChanged.connect(self.update_decay)
        self.rotate = self.findChild(QtWidgets.QCheckBox, "rotate")
        self.rotate.stateChanged.connect(self.update_rotate)
        self.save_btn = self.findChild(QtWidgets.QPushButton, "save_btn")
        self.save_btn.clicked.connect(self.export_image)
        self.save_settings_btn = self.findChild(
            QtWidgets.QPushButton, "save_settings_btn")
        self.save_settings_btn.clicked.connect(self.export_settings)
        self.neuron_checkbox = self.findChild(QtWidgets.QCheckBox, "neuron")
        self.neuron_checkbox.stateChanged.connect(self.update_neuron_settings)
        self.neuron_select = self.findChild(QtWidgets.QSlider, "neuron_select")
        self.selected_neuron = self.findChild(
            QtWidgets.QLineEdit, "selected_neuron")
        self.neuron_select.valueChanged.connect(self.update_neuron)
        self.selected_neuron.textChanged.connect(self.update_neuron)

        back_btn = self.findChild(QtWidgets.QPushButton, "back_btn")
        back_btn.clicked.connect(self.goto_main)

        generate_btn = self.findChild(QtWidgets.QPushButton, "generate_btn")
        generate_btn.clicked.connect(self.generate_visualization)

    def export_image(self):
        """Exports the generated visualization to the selected path."""
        if self.img_array is None:
            return
        path = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Image", "filter", "PNG (*.png)")[0]
        if path == "":
            return
        keras.preprocessing.image.save_img(path, self.img_array)

    def export_settings(self):
        """Exports the current feature visualization settings."""
        path = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Settings", "settings")[0]
        if path == "":
            return
        self.controller.visualizer.export_settings(path)

    def generate_visualization(self):
        """Generates and displays a visualization for the selected filter."""
        if not self.neuron_enabled:
            self.img_array = self.controller.visualizer.visualize_filter()
        else:
            self.img_array = self.controller.visualizer.visualize_neuron(
                self.neuron)
        img = Image.fromarray(self.img_array)
        qim = ImageQt(img)
        pixmap = QtGui.QPixmap.fromImage(qim)
        self.vis_container.setPixmap(pixmap.scaled(
            350, 350, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))

    def update_filter(self, value):
        """Updates the selected filter."""
        num_format = re.compile(r'^[1-9][0-9]*$')
        if not re.match(num_format, str(value)) or int(value) < 1 or int(value) > self.controller.visualizer.settings.get_layer_by_name(self.controller.visualizer.settings.layer).filter_count:
            return
        self.controller.visualizer.settings.filter = int(value) - 1
        self.selected_filter.setText(str(value))
        self.filter.setValue(int(value))

    def update_neuron_settings(self, state):
        """Updates the neuron loss function."""
        if state == QtCore.Qt.Checked:
            self.neuron_enabled = True
            self.neuron_select.setEnabled(True)
            self.selected_neuron.setEnabled(True)
            self.update_neuron(1)
        else:
            self.neuron_enabled = False
            self.neuron_select.setEnabled(False)
            self.selected_neuron.setEnabled(False)
            self.selected_neuron.setText("")

    def update_neuron(self, value):
        """Updates the selected neuron."""
        num_format = re.compile(r'^[1-9][0-9]*$')
        if not re.match(num_format, str(value)) or int(value) < 1 or int(value) > self.controller.visualizer.settings.get_layer_by_name(self.controller.visualizer.settings.layer).neuron_count:
            return
        self.neuron = int(value)
        self.selected_neuron.setText(str(value))
        self.neuron_select.setValue(int(value))

    def update_learning_rate(self, value):
        """Updates the learning rate."""
        num_format = re.compile(r"(^[0-9]*[.])?[0-9]+$")
        if not re.match(num_format, str(value)) or float(value) <= 0:
            return
        self.controller.visualizer.settings.learning_rate = float(value)

    def update_iterations(self, value):
        """Updates the number of iterations."""
        num_format = re.compile(r'^[1-9][0-9]*$')
        if not re.match(num_format, str(value)):
            return
        self.controller.visualizer.settings.iterations = int(value)

    def update_kernel(self, value):
        """Updates the kernel size used for the blur regularization."""
        num_format = re.compile(r'^[1-9][0-9]*$')
        if not re.match(num_format, str(value)):
            return
        self.controller.visualizer.settings.blur_kernel_size = int(value)

    def update_blur(self, state):
        """Updates the usage of the blur regularization."""
        if state == QtCore.Qt.Checked:
            self.controller.visualizer.settings.blur = True
        else:
            self.controller.visualizer.settings.blur = False

    def update_freq_pen(self, state):
        """Updates the usage of the frequency penalization regularization."""
        if state == QtCore.Qt.Checked:
            self.controller.visualizer.settings.freq_penalization = True
        else:
            self.controller.visualizer.settings.freq_penalization = False

    def update_decay(self, state):
        """Updates the usage of the decay regularization."""
        if state == QtCore.Qt.Checked:
            self.controller.visualizer.settings.decay = True
        else:
            self.controller.visualizer.settings.decay = False

    def update_rotate(self, state):
        """Updates the usage of the rotate regularization."""
        if state == QtCore.Qt.Checked:
            self.controller.visualizer.settings.rotate = True
        else:
            self.controller.visualizer.settings.rotate = False

    def update_layer(self, value):
        """Updates the selected layer and adjusts other setting options to match the characteristics of the new layer."""
        if value == "":
            return
        self.controller.visualizer.settings.layer = value
        self.controller.visualizer.settings.filter = 1
        filter_count = self.controller.visualizer.settings.get_layer_by_name(
            self.controller.visualizer.settings.layer).filter_count
        neuron_count = self.controller.visualizer.settings.get_layer_by_name(
            self.controller.visualizer.settings.layer).neuron_count
        self.filter.setMaximum(filter_count)
        self.neuron_select.setMaximum(neuron_count)
        self.filter.setValue(self.controller.visualizer.settings.filter)
        self.neuron_select.setValue(1)

    def update_ui(self):
        """Updates the input elements to fit the characteristics of the selected model."""
        self.choose_layer.clear()
        for layer in self.controller.visualizer.settings.conv_layers:
            self.choose_layer.addItem(layer.name)
        self.filter.setMinimum(1)
        self.neuron_select.setMinimum(1)
        if self.controller.visualizer.settings.layer is not None:
            filter_count = self.controller.visualizer.settings.get_layer_by_name(
                self.controller.visualizer.settings.layer).filter_count
            neuron_count = self.controller.visualizer.settings.get_layer_by_name(
                self.controller.visualizer.settings.layer).neuron_count
            self.filter.setMaximum(filter_count)
            self.neuron_select.setMaximum(neuron_count)
        else:
            self.controller.visualizer.settings.layer = self.choose_layer.currentText()
        self.learning_rate.setText(
            str(self.controller.visualizer.settings.learning_rate))
        self.iterations.setText(
            str(self.controller.visualizer.settings.iterations))
        self.blur.setChecked(self.controller.visualizer.settings.blur)
        if self.controller.visualizer.settings.blur:
            self.kernel.setText(
                str(self.controller.visualizer.settings.blur_kernel_size))
        self.freq_pen.setChecked(
            self.controller.visualizer.settings.freq_penalization)
        self.decay.setChecked(self.controller.visualizer.settings.decay)
        self.rotate.setChecked(self.controller.visualizer.settings.rotate)
        self.filter.setValue(self.controller.visualizer.settings.filter)
        self.selected_neuron.setText("")
        self.neuron = False
        self.neuron_select.setEnabled(False)
        self.selected_neuron.setEnabled(False)
        self.selected_filter.setText(
            str(self.controller.visualizer.settings.filter + 1))

    def goto_main(self):
        """Returns to the main menu"""
        self.controller.switch_screen(Screen.MAIN)
