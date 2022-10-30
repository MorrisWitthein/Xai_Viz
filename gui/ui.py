import sys
from backend.util import Screen
import gui.main_menu as main_menu
import gui.grouped_menu as grouped_menu
import gui.layer_rep_menu as layer_rep_menu
import gui.filter_acts_menu as filter_acts_menu
import gui.sample_menu as sample_menu
from backend.visualizer import Visualizer
from PyQt5 import QtWidgets
import qdarkstyle
import os

class Ui:
    """Main class for the application.
    Creates and initializes the available screens.
    The communication between backend and frontend is performed through the visualizer
    object created in this class.
    """

    def __init__(self):
        os.environ['QT_API'] = 'pyqt5'
        app = QtWidgets.QApplication(sys.argv)
        app.setStyleSheet(qdarkstyle.load_stylesheet())
        self.screens = QtWidgets.QStackedWidget()
        self.screens.setWindowTitle("XAI Viz")
        self.screen_width = 1280
        self.screen_height = 720
        self.screens.setFixedSize(self.screen_width, self.screen_height)
        self.init_screens()
        self.screens.show()
        self.visualizer = Visualizer()
        sys.exit(app.exec_())

    def init_screens(self):
        """Creates the available screens."""
        self.main_screen = main_menu.Main_Menu(self)
        self.sample_screen = sample_menu.Sample_Menu(self)
        self.filter_acts_screen = filter_acts_menu.Filter_Acts_Menu(self)
        self.layer_rep_screen = layer_rep_menu.Layer_Rep_Menu(self)
        self.grouped_screen = grouped_menu.Grouped_Menu(self)
        self.screens.addWidget(self.main_screen)
        self.screens.addWidget(self.sample_screen)
        self.screens.addWidget(self.filter_acts_screen)
        self.screens.addWidget(self.layer_rep_screen)
        self.screens.addWidget(self.grouped_screen)

    def switch_screen(self, new_screen):
        """Switches to the given screen. Updates the UI elements on that screen if necessary."""
        if new_screen == Screen.MAIN:
            self.screens.setCurrentWidget(self.main_screen)
        if new_screen == Screen.SAMPLE:
            self.sample_screen.update_ui()
            self.screens.setCurrentWidget(self.sample_screen)
        if new_screen == Screen.FILTER_ACTS:
            self.filter_acts_screen.update_ui()
            self.screens.setCurrentWidget(self.filter_acts_screen)
        if new_screen == Screen.LAYER_REP:
            self.layer_rep_screen.update_ui()
            self.screens.setCurrentWidget(self.layer_rep_screen)
        if new_screen == Screen.GROUPED:
            self.grouped_screen.update_ui()
            self.screens.setCurrentWidget(self.grouped_screen)

