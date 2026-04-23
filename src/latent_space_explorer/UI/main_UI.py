from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QWidget, QLabel, QComboBox, QFormLayout, QVBoxLayout, QSlider

from latent_space_explorer.custom_types import CodeSpace
from latent_space_explorer.app_manager import AppManager
from latent_space_explorer.utils.dim_reduction import DimReductionType
from latent_space_explorer.UI.selection import SelectionWidget
from latent_space_explorer.UI.interpolation import InterpolationWidget


# Direct copy of stackoverflow post of Michael Herrmann
def set_dark_theme(app, base_color: list) -> None:
    app.setStyle("Fusion")

    # Use a palette to switch to dark colors
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(*base_color))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.black)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)


class MenuWidget(QWidget):
    '''Class for containing both selection and interpolation tabs as well as menu with buttons and sliders'''

    selection_mesh_resolution_changed = pyqtSignal(int)
    interpolation_mesh_resolution_changed = pyqtSignal(int)
    rotations_per_minute_changed = pyqtSignal(int)

    def __init__(self, app_manager: AppManager):
        super().__init__()

        self.selection_mesh_resolution_changed.connect(app_manager.set_selection_resolution)
        self.interpolation_mesh_resolution_changed.connect(app_manager.set_interpolation_resolution)
        self.rotations_per_minute_changed.connect(app_manager.update_rotation_speed)

        # Create dimensionality reduction space selection GUI
        self.dim_red_label = QLabel('<b>Dimensionality reduction method</b>', self)
        self.dim_red_label.setWordWrap(True)
        self.dim_red_dropdown = QComboBox()
        for dim_reduction_type in DimReductionType:
            self.dim_red_dropdown.addItem(dim_reduction_type.name)
        self.dim_red_dropdown.setCurrentIndex(app_manager.dim_reduction_type.value)
        self.dim_red_dropdown.currentIndexChanged.connect(app_manager.update_dim_reduction_type)

        # Create pca/latent code selection GUI
        self.code_space_label = QLabel('<b>Shape code space</b>', self)  # .setWordWrap(True)
        self.code_space_dropdown = QComboBox()
        for codeSpace in CodeSpace:
            self.code_space_dropdown.addItem(codeSpace.name)
        self.code_space_dropdown.setCurrentIndex(app_manager.code_space.value)
        self.code_space_dropdown.currentIndexChanged.connect(app_manager.update_code_space)

        # Create sliders with labels
        self.select_res_label = QLabel(f'<b></br>Shape resolution:</b> {app_manager.mesh_res_selection}', self)
        self.select_res_slider = self.initialize_slider(10, 200, app_manager.mesh_res_selection, self.set_selection_resolution)

        self.interpolation_res_label = QLabel(f'<b></br>Shape resolution:</b> {app_manager.mesh_res_interpolation}', self)
        self.interpolation_res_slider = self.initialize_slider(10, 200, app_manager.mesh_res_interpolation, self.set_interpolation_resolution)

        self.rotation_speed_label = QLabel(f'<b></br>Rotations/minute:</b> {app_manager.rotations_per_minute}', self)
        self.rotation_speed_slider = self.initialize_slider(0, 20, app_manager.rotations_per_minute, self.set_rotational_speeds)

        # List widgets to be turned on and off depending on selected selection or interpolation tab
        self.selection_only_objects = [self.dim_red_label, self.dim_red_dropdown,
                                       self.code_space_label, self.code_space_dropdown,
                                       self.select_res_label, self.select_res_slider]
        self.interpolation_only_objects = [self.interpolation_res_label, self.interpolation_res_slider]
        self.tab_specific_objects_lists = [self.selection_only_objects, self.interpolation_only_objects]

        # Create a container widget for QFormLayout content_layout
        layout_container = QWidget()
        content_layout = QFormLayout(layout_container)
        # content_layout.setSpacing(0)

        # Add dropdown for code space and dimensionality reduction selection
        content_layout.addRow(self.dim_red_label)
        content_layout.addRow(self.dim_red_dropdown)
        self.add_vertical_space(10, content_layout)

        content_layout.addRow(self.code_space_label)
        content_layout.addRow(self.code_space_dropdown)
        self.add_vertical_space(10, content_layout)

        # Add sliders and labels to the form layout
        self.add_label_slider_UI(self.select_res_slider, self.select_res_label, content_layout)
        self.add_label_slider_UI(self.interpolation_res_slider, self.interpolation_res_label, content_layout)
        self.add_label_slider_UI(self.rotation_speed_slider, self.rotation_speed_label, content_layout)

        main_layout = QVBoxLayout(self)
        main_layout.addStretch()
        main_layout.addWidget(layout_container)
        main_layout.addStretch()

    def initialize_slider(self, min: int, max: int, initial: int, function) -> QSlider:
        slider = QSlider(Qt.Horizontal, self)
        slider.setMinimum(min)
        slider.setMaximum(max)
        slider.setValue(initial)
        slider.valueChanged[int].connect(function)
        return slider

    def add_label_slider_UI(self, slider: QSlider, label: QLabel, parent_layout: any) -> QFormLayout:
        slider_label_UI = QFormLayout()
        slider_label_UI.setSpacing(0)
        slider_label_UI.addRow(label)
        slider_label_UI.addRow(slider)
        parent_layout.addRow(slider_label_UI)
        return slider_label_UI

    def set_selection_resolution(self, slider_value: int) -> None:
        self.select_res_label.setText(f'<b>Shape resolution:</b> {slider_value}')
        self.selection_mesh_resolution_changed.emit(slider_value)

    def set_interpolation_resolution(self, slider_value: int) -> None:
        self.interpolation_res_label.setText(f'<b>Shape resolution:</b> {slider_value}')
        self.interpolation_mesh_resolution_changed.emit(slider_value)

    def set_rotational_speeds(self, slider_value: int) -> None:
        self.rotation_speed_label.setText(f'<b>Rotations/minute:</b> {slider_value}')
        self.rotations_per_minute_changed.emit(slider_value)

    def add_vertical_space(self, spacing: int, parent_layout) -> None:
        parent_layout.addItem(QtWidgets.QSpacerItem(0, spacing,
                                                    QtWidgets.QSizePolicy.Minimum,
                                                    QtWidgets.QSizePolicy.Fixed))


class MainWindow(QtWidgets.QWidget):
    def __init__(self,
                 selection_tab: SelectionWidget, interpolation_tab: InterpolationWidget,
                 menu: MenuWidget, app_manager: AppManager):
        super().__init__()

        self.menu = menu
        self.selection_tab = selection_tab
        self.tab_interpolation = interpolation_tab

        self.setWindowTitle("Latent space explorer")

        # Initialize tabs widget and add tabs
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.selection_tab, "Code selection")
        self.tabs.addTab(self.tab_interpolation, "Shape interpolation")
        self.tabs.currentChanged.connect(self.changed_tab)
        self.tabs.currentChanged.connect(app_manager.changed_tab)
        self.changed_tab(0)

        self.menu.setFixedWidth(200)

        self.layout = QtWidgets.QHBoxLayout()
        self.layout.addWidget(self.tabs)
        self.layout.addWidget(self.menu)
        self.setLayout(self.layout)

        self.resize(1500, 800)
        self.show()

    def changed_tab(self, idx_tab: int) -> None:
        """Adjust UI menu when detecting tab change"""
        for idx_tab_specific_objs_list, tab_specific_objects_list in enumerate(self.menu.tab_specific_objects_lists):
            for widget in tab_specific_objects_list:
                widget.setVisible(idx_tab_specific_objs_list == idx_tab)
