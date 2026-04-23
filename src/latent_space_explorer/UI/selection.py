from PyQt5 import QtWidgets

from latent_space_explorer.app_manager import AppManager
from latent_space_explorer.UI.dim_reduction import DimReductionWidget
from latent_space_explorer.UI.latent_code import CodeImgWidget
from latent_space_explorer.UI.mesh import MeshWidget


class SelectionWidget(QtWidgets.QWidget):
    '''Selection tab class for holding reduced dimensionality, latent code and shape visualisation widgets'''
    def __init__(self,
                 dim_reduction_widget: DimReductionWidget, code_img_widget: CodeImgWidget,
                 first_mesh_widget: MeshWidget, secondary_mesh_widget: MeshWidget,
                 app_manager: AppManager):
        super().__init__()

        app_manager.visualize_mode_changed.connect(self.adjust_visualize_mode)

        self.visualise_state = 0

        self.first_mesh_widget = first_mesh_widget
        self.secondary_mesh_widget = secondary_mesh_widget

        # self.showFullScreen()
        self.resize(1400, 800)

        # Create grid layout and widgets and place widgets in grid
        self.grid_layout = QtWidgets.QGridLayout(self)

        self.grid_layout.addWidget(dim_reduction_widget, 0, 0)
        self.grid_layout.addWidget(code_img_widget, 1, 0)
        self.grid_layout.addWidget(self.first_mesh_widget, 0, 1, 2, 1)  # Initially spanning 2 rows
        self.grid_layout.addWidget(self.secondary_mesh_widget, 1, 1, 1, 1)
        self.secondary_mesh_widget.hide()

    def adjust_visualize_mode(self, visualise_state: int) -> None:
        if visualise_state == 0:
            # Hide 2nd/bottom widget and make 1st/top widget larger
            self.secondary_mesh_widget.hide()
            self.grid_layout.removeWidget(self.first_mesh_widget)
            self.grid_layout.addWidget(self.first_mesh_widget, 0, 1, 2, 1)
            self.first_mesh_widget.show()
        else:
            # Crimp 1st/top widget and "add" 2nd/bottom
            self.grid_layout.removeWidget(self.first_mesh_widget)
            self.grid_layout.addWidget(self.first_mesh_widget, 0, 1, 1, 1)  # Top
            self.secondary_mesh_widget.show()  # Bottom
