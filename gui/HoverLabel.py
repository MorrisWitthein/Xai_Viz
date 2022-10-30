from PyQt5.QtCore import pyqtSignal
from PyQt5 import QtWidgets


class HoverLabel(QtWidgets.QLabel):
    """Custom implementation of the QLabel.
    This implementation adds mouse tracking that is used to display highlights on generated visualizations.
    """
    mouseHover = pyqtSignal(bool)

    def __init__(self, parent=None):
        QtWidgets.QLabel.__init__(self, parent)
        self.setMouseTracking(True)
        self.event = None

    def mouseMoveEvent(self, event):
        self.event = event
        self.mouseHover.emit(True)

    def leaveEvent(self, event):
        self.mouseHover.emit(False)
