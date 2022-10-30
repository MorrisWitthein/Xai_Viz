from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QThread
import os
from gui.worker import Worker
from enum import Enum
import backend.util as util


class Generate_Popup(QtWidgets.QDialog):
    """GUI class for the Popup that is displayed whenever a long generation task is in progress."""

    def __init__(self, controller, task, target=None, path=None):
        super(Generate_Popup, self).__init__()
        path = util.resource_path(os.path.join('res', 'ui', 'generate.ui'))
        uic.loadUi(path, self)
        self.controller = controller
        self.target = target
        self.task = task
        self.setFixedSize(400, 300)

        self.label = self.findChild(QtWidgets.QLabel, "label")
        self.cancel_btn = self.findChild(QtWidgets.QPushButton, "cancel_btn")
        self.cancel_btn.clicked.connect(self.cancel)
        self.progress_bar = self.findChild(
            QtWidgets.QProgressBar, "progress_bar")

        if self.task == Task.DICTIONARY:
            self.setWindowTitle("Generate Dictionary")
            self.progress_bar.setRange(
                0, len(self.controller.visualizer.settings.conv_layers))
        elif self.task == Task.LAYER_REP:
            self.setWindowTitle("Generate Layer Representation")
            self.label.setText("Generating Visualization...")
            self.progress_bar.setRange(0, len(
                self.controller.visualizer.activations) * len(self.controller.visualizer.activations))
        elif self.task == Task.GROUPS:
            self.setWindowTitle("Generate Group Visualizations")
            self.label.setText("Generating Visualization...")
            self.progress_bar.setRange(
                0, self.controller.visualizer.settings.groups)
        self.start()

    def start(self):
        """Creates a new thread and starts the selected task."""
        self.thread = QThread()
        self.worker = Worker(self.controller)
        self.worker.moveToThread(self.thread)
        if self.task == Task.DICTIONARY:
            self.thread.started.connect(self.worker.run_generate_dictionary)
        elif self.task == Task.LAYER_REP:
            self.thread.started.connect(
                self.worker.run_generate_activation_grid)
        elif self.task == Task.GROUPS:
            self.thread.started.connect(
                self.worker.run_generate_group_visualization)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.finish)
        self.worker.progress.connect(self.reportProgress)
        self.thread.start()

    def cancel(self):
        """When the cancel Button is clicked the current task is canceled and the thread is stopped."""
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()
        # If the task was dictionary generation reset the dictionary since only complete dictionaries are valid.
        if self.task == Task.DICTIONARY:
            self.controller.visualizer.reset_dictionary()
            self.controller.visualizer.settings.dict_path = None
        self.close()

    def finish(self):
        """Stops the thread when a task is completed."""
        self.close()
        self.thread.deleteLater

    def reportProgress(self, n):
        """Updates the displayed progress bar."""
        self.progress_bar.setValue(n)

    def closeEvent(self, event):
        """Routine that gets called when the popup is closed. Makes sure the thread is stopped correctly before the window is closed."""
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()
        if not self.worker.completed and self.task == Task.DICTIONARY:
            self.controller.visualizer.reset_dictionary()
            self.controller.visualizer.settings.dict_path = None
        event.accept()


class Task(str, Enum):
    """Enumeration to distinguish between the different tasks that can be executed using this popup."""
    DICTIONARY = "DICTIONARY"
    LAYER_REP = "LAYER_REP"
    GROUPS = "GROUPS"
