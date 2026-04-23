import os
import sys
from PyQt5 import QtWidgets # TODO update to official wrapper pyside or PyQt6

from latent_space_explorer.config import Config
from latent_space_explorer.custom_types import ShapeType
from latent_space_explorer.app_manager import AppManager
from latent_space_explorer.UI.main_UI import set_dark_theme, MainWindow, MenuWidget
from latent_space_explorer.UI.selection import SelectionWidget
from latent_space_explorer.UI.interpolation import InterpolationWidget
from latent_space_explorer.UI.dim_reduction import DimReductionWidget
from latent_space_explorer.UI.latent_code import CodeImgWidget
from latent_space_explorer.UI.mesh import MeshWidget

if __name__ == '__main__':
    config = Config()
    os.makedirs(config.dim_reduction_textures_dir, exist_ok = True)
    os.makedirs(config.reconstructed_meshes_dir, exist_ok = True)

    app = QtWidgets.QApplication(sys.argv)
    set_dark_theme(app, config.base_color)

    app_manager = AppManager(config)

    # Create selection widget children
    dimReductionWidget = DimReductionWidget(app_manager)
    code_img_widget = CodeImgWidget(app_manager = app_manager)
    single_shape_select_widget = MeshWidget(config.mesh_paths[ShapeType.SELECT_1], app_manager)
    second_shape_select_widget = MeshWidget(config.mesh_paths[ShapeType.SELECT_2], app_manager)
    
    selection_tab = SelectionWidget(dimReductionWidget, code_img_widget, single_shape_select_widget, second_shape_select_widget, app_manager)
    interpolation_tab = InterpolationWidget(config.mesh_paths[ShapeType.INTERPOLATE], app_manager)
    menu = MenuWidget(app_manager)

    main_window = MainWindow(selection_tab, interpolation_tab, menu, app_manager)

    sys.exit(app.exec_())