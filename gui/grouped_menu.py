from PyQt5 import QtWidgets, QtGui, uic, QtCore
from PIL.ImageQt import ImageQt
import tensorflow as tf
import os
import math

from gui.generate_popup import Generate_Popup, Task
from gui.HoverLabel import HoverLabel
import backend.util as util
from backend.util import Screen


class Grouped_Menu(QtWidgets.QWidget):
    """GUI class that generates groups of activations and displays activation maps 
    and feature visualizations for each of these groups.
    """

    def __init__(self, controller):
        super(Grouped_Menu, self).__init__()
        self.controller = controller
        path = util.resource_path(os.path.join('res', 'ui', 'grouped.ui'))
        uic.loadUi(path, self)
        back_btn = self.findChild(QtWidgets.QPushButton, "back_btn")
        back_btn.clicked.connect(self.goto_back)

        back_btn = self.findChild(QtWidgets.QPushButton, "back_btn")
        back_btn.clicked.connect(self.goto_back)

        generate_btn = self.findChild(QtWidgets.QPushButton, "generate_btn")
        generate_btn.clicked.connect(self.generate_visualization)

        self.target_choice = self.findChild(
            QtWidgets.QCheckBox, "target_choice")
        self.grad_cam = self.findChild(QtWidgets.QCheckBox, "grad_cam")

        self.group_slider = self.findChild(QtWidgets.QSlider, "group_slider")
        self.group_slider.valueChanged.connect(self.update_groups)

        self.init_vis_container()

        self.input_highlight = []
        self.vis_highlight = []

        self.choose_layer = self.findChild(QtWidgets.QComboBox, "choose_layer")
        self.choose_layer.currentTextChanged.connect(self.update_layer)

        self.save_btn = self.findChild(QtWidgets.QPushButton, "save_btn")
        self.save_btn.clicked.connect(self.export_image)

    def init_vis_container(self):
        """Initializes the QLabels that act as a container for the generated feature visualizations."""
        self.group_vis_container = []
        #self.group_attributions =[]
        for i in range(1, 7):
            self.group_vis_container.append(
                self.findChild(QtWidgets.QLabel, f"group{i}"))
            #self.group_attributions.append(self.findChild(QtWidgets.QLabel, f"attr{i}"))
            self.group_vis_container[-1].mousePressEvent = self.update_visualization

    def update_ui(self):
        """When the user selects an different layer the current visualization and highlights are reset.
        Additionally the visualization for the new layer are imported from the harddrive.
        """
        self.generated = False
        self.group_label = self.findChild(QtWidgets.QLabel, "groups")
        self.group_label.setText(
            f"{self.controller.visualizer.settings.groups} groups")
        self.group_slider.setValue(
            self.controller.visualizer.settings.groups / 2)
        self.input_container = HoverLabel(self)
        self.vis_container = HoverLabel(self)
        for i in range(len(self.group_vis_container)):
            self.group_vis_container[i].clear()
            self.group_vis_container[i].setStyleSheet("border: ")
            # self.group_attributions[i].setText("")
        self.img_size = self.controller.visualizer.settings.input_width * 2
        self.input_container.setFixedSize(self.img_size, self.img_size)
        self.vis_container.setFixedSize(self.img_size, self.img_size)
        self.input_container.move(
            self.controller.screen_width // 3 - self.img_size // 2, 100)
        self.vis_container.move(self.input_container.x() + self.img_size, 100)
        self.update_input_image(self.controller.visualizer.settings.input_img)
        self.choose_layer.clear()
        for layer in self.controller.visualizer.settings.grp_layers:
            if not "out" in (layer.name) or layer.out_shape[1] > self.controller.visualizer.settings.input_width:
                continue
            self.choose_layer.addItem(layer.name)
        self.input_container.mouseHover.connect(self.update_input_highlight)
        self.vis_container.mouseHover.connect(self.update_vis_highlight)

    def update_groups(self, value):
        """Updates the label displaying how many groups are selected."""
        self.group_label.setText(f"{int(value) * 2} groups")

    def update_input_image(self, img):
        """Updates the display inout image."""
        self.in_qim = ImageQt(img)
        self.in_pixmap = QtGui.QPixmap.fromImage(self.in_qim)
        self.input_container.setPixmap(self.in_pixmap.scaled(
            self.img_size, self.img_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))

    def export_image(self):
        """Exports the selected activation map and the corresponding feature visualization."""
        if not self.generated:
            return
        path = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Image", "layer", "PNG (*.png)")
        if path[0] == "":
            return
        export = util.combine_group_img(self.controller.visualizer.settings,
                                        self.group_activation_maps[self.selected_group], self.grp_imgs[self.selected_group])
        tf.keras.utils.save_img(path[0], export)

    def generate_visualization(self):
        """Starts the task to generate the group visualizations on a new thread.
        Displays the progress popup during the process.
        """
        self.popup = Generate_Popup(self.controller, Task.GROUPS)
        self.popup.setWindowFlags(
            self.popup.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)
        self.popup.exec_()

    def update_visualization(self, event):
        """Updates the displayed activation map when a different activation group is selected."""
        self.calculate_selected_group(event.windowPos().x())
        if self.selected_group < self.controller.visualizer.settings.groups:
            for label in self.group_vis_container:
                label.setStyleSheet("border: ")
            self.group_vis_container[self.selected_group].setStyleSheet(
                "border: 2px solid orange")
            self.vis_container.clear()
            self.vis_qim = ImageQt(
                self.group_activation_maps[self.selected_group])
            self.vis_pixmap = QtGui.QPixmap.fromImage(self.vis_qim)
            self.vis_container.setPixmap(self.vis_pixmap)

    def update_layer(self, value):
        """When the user selects an different layer the current visualization and highlights are reset."""
        self.remove_prev(delete_all=True)
        if value == "":
            return
        self.controller.visualizer.settings.layer = value
        self.controller.visualizer.update_activations()
        self.vis_container.clear()
        self.generated = False
        self.input_grid = []
        self.vis_grid = []
        for i in range(len(self.group_vis_container)):
            self.group_vis_container[i].clear()
            self.group_vis_container[i].setStyleSheet("border: ")
        input_shape = self.controller.visualizer.settings.input_width
        layer = self.controller.visualizer.settings.get_layer_by_name(value)
        self.size = 2 * (input_shape // layer.out_shape[1])
        for i in range(layer.out_shape[1]):
            self.input_grid.append([])
            self.vis_grid.append([])
            for j in range(layer.out_shape[2]):
                self.input_grid[i].append((self.input_container.x(
                ) + i * self.size, self.input_container.y() + j * self.size))
                self.vis_grid[i].append(
                    (self.vis_container.x() + i * self.size, self.vis_container.y() + j * self.size))

    def update_input_highlight(self, value, vis_grid_pos=None, rec=False):
        """Displays the highlight on the input image when the mouse is moved on the displayed input image.
        Also calls the update function for the highlight on the activation map if it was generated yet.
        """
        grid_pos = None
        if value:
            if vis_grid_pos is None:
                grid_pos = self.get_grid_position(self.sender().event)
            else:
                grid_pos = vis_grid_pos
            x = grid_pos[0]
            y = grid_pos[1]
            self.opacity_effect = QtWidgets.QGraphicsOpacityEffect()
            self.opacity_effect.setOpacity(0.3)
            self.input_highlight.append(QtWidgets.QLabel(self))
            self.input_highlight[-1].setStyleSheet("background-color: orange")
            self.input_highlight[-1].setGeometry(
                self.input_grid[x][y][0], self.input_grid[x][y][1], self.size, self.size)
            self.input_highlight[-1].setGraphicsEffect(self.opacity_effect)
            self.input_highlight[-1].show()
        else:
            self.remove_prev()
        if not rec:
            self.update_vis_highlight(value, grid_pos, True)

    def update_vis_highlight(self, value, input_grid_pos=None, rec=False):
        """Displays the highlight on the activation map when the mouse is moved on the displayed visualization.
        Also calls the update function for the highlight on the input image.
        """
        if not self.generated:
            return
        grid_pos = None
        if value:
            if input_grid_pos is None:
                grid_pos = self.get_grid_position(self.sender().event)
            else:
                grid_pos = input_grid_pos
            x = grid_pos[0]
            y = grid_pos[1]
            self.opacity_effect = QtWidgets.QGraphicsOpacityEffect()
            self.opacity_effect.setOpacity(0.3)
            self.vis_highlight.append(QtWidgets.QLabel(self))
            self.vis_highlight[-1].setStyleSheet("background-color: orange")
            self.vis_highlight[-1].setGeometry(
                self.vis_grid[x][y][0], self.vis_grid[x][y][1], self.size, self.size)
            self.vis_highlight[-1].setGraphicsEffect(self.opacity_effect)
            self.vis_highlight[-1].show()
        else:
            self.remove_prev()
        if not rec:
            self.update_input_highlight(value, grid_pos, True)

    def remove_prev(self, delete_all=False):
        """Removes previous highlights from the list. Clears all highlights if delete_all is set."""
        while len(self.input_highlight) > 1:
            self.input_highlight[0].deleteLater()
            self.input_highlight.remove(self.input_highlight[0])
        while len(self.vis_highlight) > 1:
            self.vis_highlight[0].deleteLater()
            self.vis_highlight.remove(self.vis_highlight[0])
        if delete_all:
            if len(self.vis_highlight) > 0:
                self.vis_highlight[0].deleteLater()
            if len(self.input_highlight) > 0:
                self.input_highlight[0].deleteLater()
            self.vis_highlight = []
            self.input_highlight = []

    def get_grid_position(self, mouse_position):
        """Converts the mouse position on the displayed images to a coordinate.
        This coordinate is used to determine which feature visualization is displayed
        in the magnified view underneath the layer representation.
        """
        x = mouse_position.x() // self.size
        y = mouse_position.y() // self.size
        return (x, y)

    def calculate_selected_group(self, x):
        """Calculates which group was selected by clicking on the corresponding feature visualization depending on the mouse position."""
        self.selected_group = math.floor((x - 270) / 130)

    def goto_back(self):
        """Switches back to the main menu."""
        self.remove_prev(True)
        self.controller.switch_screen(Screen.MAIN)
