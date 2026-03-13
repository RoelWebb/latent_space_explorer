from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSignal

from latent_space_explorer.app_manager import AppManager
from latent_space_explorer.UI import mesh

class InterpolationWidget(QtWidgets.QWidget):
    interpolation_changed = pyqtSignal(int)

    def __init__(self, shape_path : str, app_manager : AppManager):
        super().__init__()

        app_manager.interpolation_tab_activated.connect(self.reset_interpolation_UI)

        shape_widget = mesh.MeshWidget(shape_path, app_manager)

        # Create slider widget
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(0)
        self.slider.valueChanged[int].connect(app_manager.linear_interpolate)

        # Initialize layout and add shape_widget and slider
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(shape_widget)
        self.layout.addWidget(self.slider)
        self.setLayout(self.layout)
        
    def reset_interpolation_UI(self) -> None:
        """Set slider back to 0 position"""
        self.slider.setValue(0)