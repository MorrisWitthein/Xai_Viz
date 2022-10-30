from backend.util import Screen
from PyQt5 import QtWidgets, QtGui, uic
from PyQt5.QtCore import Qt
import os

import backend.util as util
from gui.generate_popup import Generate_Popup, Task


class Main_Menu(QtWidgets.QWidget):
    """GUI class for the Main menu.
    Used to navigate between the different screens.
    Also provides functionalities for importing models, inputs and dictionaries
    with filter visualizations.
    """

    def __init__(self, controller):
        super(Main_Menu, self).__init__()
        self.controller = controller
        self.setWindowTitle("XAI Viz")
        path = util.resource_path(os.path.join('res', 'ui', 'main_window.ui'))
        uic.loadUi(path, self)
        self.loading_lbl = self.findChild(QtWidgets.QLabel, "loading_lbl")
        self.buttons = dict()
        self.init_buttons()
        self.input = False
        self.dictionary = False

    def init_buttons(self):
        """Imports the UI elements from the ui file and connects them with their corresponding functions."""
        self.buttons["load_model"] = self.findChild(
            QtWidgets.QPushButton, "load_model_btn")
        self.buttons["load_model"].clicked.connect(self.load_network)

        self.buttons["load_input"] = self.findChild(
            QtWidgets.QPushButton, "load_input_btn")
        self.buttons["load_input"].clicked.connect(self.load_input)

        self.buttons["sample"] = self.findChild(
            QtWidgets.QPushButton, "sample_btn")
        self.buttons["sample"].clicked.connect(self.goto_sample)

        self.buttons["filter_acts"] = self.findChild(
            QtWidgets.QPushButton, "filter_acts_btn")
        self.buttons["filter_acts"].clicked.connect(self.goto_filter_acts)

        self.buttons["layer_rep"] = self.findChild(
            QtWidgets.QPushButton, "layer_rep_btn")
        self.buttons["layer_rep"] .clicked.connect(self.goto_layer_rep)

        self.buttons["grouped"] = self.findChild(
            QtWidgets.QPushButton, "grouped_btn")
        self.buttons["grouped"].clicked.connect(self.goto_grouped)

        self.buttons["generate"] = self.findChild(
            QtWidgets.QPushButton, "generate_btn")
        self.buttons["generate"].clicked.connect(self.export_dictionary)

        self.buttons["import"] = self.findChild(
            QtWidgets.QPushButton, "import_btn")
        self.buttons["import"].clicked.connect(self.import_dictionary)

        self.buttons["import_sett"] = self.findChild(
            QtWidgets.QPushButton, "import_settings_btn")
        self.buttons["import_sett"].clicked.connect(self.import_settings)

    def load_network(self):
        """Tries to load a Keras model from the given path.
        Initializes model information in the backend on success.
        """
        self.loading_lbl.setVisible(True)
        path = QtWidgets.QFileDialog.getExistingDirectory()
        if path == "":
            self.loading_lbl.setVisible(False)
            return
        try:
            model = util.load_model(path)
            self.controller.visualizer.settings.init_model(model)
            self.show_checkmark(True)
            self.enable_model_buttons()
        except (TypeError, OSError) as err:
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("Error")
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText(repr(err))
            msg.exec_()

        self.loading_lbl.setVisible(False)

    def load_input(self):
        """Tries to load an input image from the given path."""
        path = QtWidgets.QFileDialog.getOpenFileName()[0]
        if path == "":
            return
        try:
            self.controller.visualizer.update_input(path)
            self.show_checkmark(False)
            self.input = True
            self.enable_input_buttons()
        except (TypeError, OSError) as err:
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("Error")
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText(repr(err))
            msg.exec_()

    def import_settings(self):
        """Tries to load a settings file from the given path."""
        path = QtWidgets.QFileDialog.getOpenFileName()[0]
        if path == "":
            return
        try:
            self.controller.visualizer.update_settings(path)
        except (TypeError, OSError) as err:
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("Error")
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText(repr(err))
            msg.exec_()

    def generate_dictionary(self, path):
        """Starts the process for generating visualizations for each filter in the 
        convolution layers of the network. Displays the progress popup during the process.
        """
        self.popup = Generate_Popup(
            self.controller, Task.DICTIONARY, path=path)
        self.popup.setWindowFlags(
            self.popup.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.popup.exec_()
        if len(self.controller.visualizer.dictionary.dictionary) > 0:
            self.dictionary = True
            self.enable_input_buttons()

    def export_dictionary(self):
        """Exports the generated filter visualizations to the given path."""
        path = QtWidgets.QFileDialog.getExistingDirectory()
        try:
            self.controller.visualizer.settings.set_dict_path(path)
            self.generate_dictionary(path)
        except (TypeError, OSError) as err:
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("Error")
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText(repr(err))
            msg.exec_()

    def import_dictionary(self):
        """Tries to import a filter visualization dictionary from the given path."""
        path = QtWidgets.QFileDialog.getExistingDirectory()
        if path == "":
            return
        try:
            self.controller.visualizer.dictionary.import_dictionary(
                path, self.controller.visualizer.settings.conv_layers[0].name)
            self.dictionary = True
            self.controller.visualizer.settings.set_dict_path(path)
            self.enable_input_buttons()
        except (TypeError, OSError) as err:
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("Error")
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText(repr(err))
            msg.exec_()

    def enable_model_buttons(self):
        """Enables buttons for actions that can only be performed after a model is imported successfully."""
        self.buttons["sample"].setEnabled(True)
        self.buttons["load_input"].setEnabled(True)
        self.buttons["generate"].setEnabled(True)
        self.buttons["import"].setEnabled(True)
        self.buttons["import_sett"].setEnabled(True)

    def enable_input_buttons(self):
        """Enables buttons for actions that can only be performed after an input image is imported successfully."""
        if self.input:
            self.buttons["grouped"].setEnabled(True)
            if self.dictionary:
                self.buttons["filter_acts"].setEnabled(True)
                self.buttons["layer_rep"].setEnabled(True)

    def show_checkmark(self, model):
        """Displays a checkmark to give the user feedback for successfully importing a model or an input image."""
        label = QtWidgets.QLabel(self)
        path = util.resource_path(os.path.join("res", "checkmark.png"))
        pixmap = QtGui.QPixmap(path)

        label.setPixmap(pixmap.scaled(
            32, 32, Qt.KeepAspectRatio, Qt.FastTransformation))

        if model:
            label.move(360, 270)
        else:
            label.move(360, 390)
        label.show()

    def goto_sample(self):
        """Switches to the sample screen."""
        self.controller.switch_screen(Screen.SAMPLE)

    def goto_filter_acts(self):
        """Switches to the filter activation investigation screen."""
        self.controller.switch_screen(Screen.FILTER_ACTS)

    def goto_layer_rep(self):
        """Switches to the layer representation screen."""
        self.controller.switch_screen(Screen.LAYER_REP)

    def goto_grouped(self):
        """Switches to the layer representation screen using grouped activations."""
        self.controller.switch_screen(Screen.GROUPED)
