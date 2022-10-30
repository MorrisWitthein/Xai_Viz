from PyQt5 import QtWidgets, QtGui, uic, QtCore
from PIL.ImageQt import ImageQt
import os

from gui.HoverLabel import HoverLabel
import backend.util as util
from backend.util import Screen


class Filter_Acts_Menu(QtWidgets.QWidget):
    """GUI class for the filter activation screen.
    Displays feature visualizations for the most activated filters of the hovered location.
    """

    def __init__(self, controller):
        super(Filter_Acts_Menu, self).__init__()
        self.controller = controller
        path = util.resource_path(os.path.join(
            'res', 'ui', 'filter_activation_screen.ui'))
        uic.loadUi(path, self)
        self.filter_container = []
        self.init_filter_container()
        self.highlight = []

        self.filterA = self.findChild(QtWidgets.QLabel, "filterAName")
        self.filterB = self.findChild(QtWidgets.QLabel, "filterBName")
        self.filterC = self.findChild(QtWidgets.QLabel, "filterCName")

        back_btn = self.findChild(QtWidgets.QPushButton, "back_btn")
        back_btn.clicked.connect(self.goto_back)

        self.choose_layer = self.findChild(QtWidgets.QComboBox, "choose_layer")
        self.choose_layer.currentTextChanged.connect(self.update_layer)

    def init_filter_container(self):
        """Initializes the labels that act as containers for the feature visualizations."""
        self.filter_container.append(
            self.findChild(QtWidgets.QLabel, "filterA"))
        self.filter_container.append(
            self.findChild(QtWidgets.QLabel, "filterB"))
        self.filter_container.append(
            self.findChild(QtWidgets.QLabel, "filterC"))

    def update_layer(self, value):
        """When the user selects an different layer the current visualization and highlights are reset.
        Additionally the visualization for the new layer are imported from the harddrive.
        """
        if value == "":
            return
        self.controller.visualizer.settings.layer = value
        self.controller.visualizer.update_activations()
        print(self.controller.visualizer.settings.dict_path)
        self.controller.visualizer.update_dictionary(value)
        self.grid = []
        input_shape = self.controller.visualizer.settings.input_width
        layer = self.controller.visualizer.settings.get_layer_by_name(value)
        self.size = 2 * (input_shape // layer.out_shape[1])
        for i in range(layer.out_shape[1]):
            self.grid.append([])
            for j in range(layer.out_shape[2]):
                self.grid[i].append(
                    (self.vis_container.x() + i * self.size, self.vis_container.y() + j * self.size))
        for label in self.highlight:
            label.deleteLater()
        self.highlight = []

    def update_ui(self):
        """Resets the UI elements of the screen when it is visited."""
        self.vis_container = HoverLabel(self)
        self.img_size = self.controller.visualizer.settings.input_width * 2
        self.vis_container.setFixedSize(self.img_size, self.img_size)
        self.vis_container.move(
            self.controller.screen_width // 2 - self.img_size // 2, 100)
        qim = ImageQt(self.controller.visualizer.settings.input_img)
        pixmap = QtGui.QPixmap.fromImage(qim)
        self.vis_container.setPixmap(pixmap.scaled(
            self.img_size, self.img_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))
        self.choose_layer.clear()
        for layer in self.controller.visualizer.settings.conv_layers:
            if layer.out_shape[1] > self.controller.visualizer.settings.input_width:
                continue
            self.choose_layer.addItem(layer.name)
        self.vis_container.mouseHover.connect(self.update_visualization)

    def update_visualization(self, value):
        """Displays the feature visualizations for the top three activated filters at the position the mouse is currently hovering over."""
        if value:
            grid_pos = self.get_grid_position(self.sender().event)
            self.update_highlight(grid_pos[0], grid_pos[1])
            filters = self.controller.visualizer.get_filter_visualizations(
                grid_pos[0], grid_pos[1])
            imgs = filters[0]
            filter_idx = filters[1]
            self.filterA.setText(f"Filter{filter_idx[0]}")
            self.filterB.setText(f"Filter{filter_idx[1]}")
            self.filterC.setText(f"Filter{filter_idx[2]}")
            for i, label in enumerate(self.filter_container):
                qim = ImageQt(imgs[i])
                pixmap = QtGui.QPixmap.fromImage(qim)
                label.setPixmap(pixmap.scaled(
                    100, 100, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))
        else:
            self.remove_prev()

    def update_highlight(self, x, y):
        """Displays the highlight on the input image that shows which position is currently howered over."""
        self.opacity_effect = QtWidgets.QGraphicsOpacityEffect()
        self.opacity_effect.setOpacity(0.3)
        self.highlight.append(QtWidgets.QLabel(self))
        self.highlight[-1].setStyleSheet("background-color: orange")
        self.highlight[-1].setGeometry(self.grid[x][y]
                                       [0], self.grid[x][y][1], self.size, self.size)
        self.highlight[-1].setGraphicsEffect(self.opacity_effect)
        self.highlight[-1].show()

    def remove_prev(self, delete_all=False):
        """Removes previous highlights from the list. Clears all highlights if delete_all is set."""
        while len(self.highlight) > 1:
            self.highlight[0].deleteLater()
            self.highlight.remove(self.highlight[0])
        if delete_all:
            if len(self.highlight) > 0:
                self.highlight[0].deleteLater()
            self.highlight = []

    def get_grid_position(self, mouse_position):
        """Converts the mouse position on the displayed images to a coordinate.
        This coordinate is used to determine which feature visualization is displayed.
        """
        x = mouse_position.x() // self.size
        y = mouse_position.y() // self.size
        return (x, y)

    def goto_back(self):
        """Switches back to the main menu."""
        self.controller.switch_screen(Screen.MAIN)
