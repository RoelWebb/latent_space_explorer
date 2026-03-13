import os
import sys
from PyQt5 import QtWidgets # TODO potentially update to PyQt6

project_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
package_path = os.path.join(project_path, 'src/latent_space_explorer')
# Make latent_space_explorer package visible to run from project dir
sys.path.insert(0, os.path.dirname(package_path))

from latent_space_explorer.custom_types import ShapeType
from latent_space_explorer.app_manager import AppManager
from latent_space_explorer.utils.dim_reduction import DimReductionType
from latent_space_explorer.UI.main_UI import set_dark_theme, MainWindow, MenuWidget
from latent_space_explorer.UI.selection import SelectionWidget
from latent_space_explorer.UI.interpolation import InterpolationWidget
from latent_space_explorer.UI.dim_reduction import DimReductionWidget
from latent_space_explorer.UI.latent_code import CodeImgWidget
from latent_space_explorer.UI.mesh import MeshWidget

category = 'cars_vessels_airplanes'

device = 'cuda:0'

# Visualization parameters
perplexity = 30
mesh_res_selection = 100
mesh_res_interpolation = 64
base_color = (25, 25, 25) # 0-255

experiment_path = os.path.join(project_path, "experiments", category)

# NN paths
model_path = os.path.join(project_path, 'DeepSDF')
specs_path = os.path.join(experiment_path, 'specs.json')
model_params_path = os.path.join(experiment_path, 'ModelParameters/latest.pth')
latent_codes_path = os.path.join(experiment_path, 'LatentCodes/latest.pth')
split_path = os.path.join(project_path, 'splits', f'{category}_train.json')
split_dataset_name = 'ShapeNetV2'

shaders_dir_path = os.path.join(package_path, 'shaders')

# Output paths
out_dir = os.path.join(experiment_path, 'out')
dim_reduction_textures_dir = os.path.join(out_dir, 'dim_reduction_textures')
reconstructed_meshes_dir = os.path.join(out_dir, 'reconstructed_meshes')
plot_dim_reduction_paths = {dim_reduction_type :
                            os.path.join(dim_reduction_textures_dir, f'{dim_reduction_type.name}_plot.png') for dim_reduction_type in DimReductionType}
mesh_paths = {shape_type : os.path.join(reconstructed_meshes_dir, f'{shape_type.name}.ply') for shape_type in ShapeType}
os.makedirs(dim_reduction_textures_dir, exist_ok = True)
os.makedirs(reconstructed_meshes_dir, exist_ok = True)

if __name__ == '__main__':    
    app = QtWidgets.QApplication(sys.argv)
    set_dark_theme(app, base_color)

    app_manager = AppManager(device,
                      perplexity, mesh_res_selection, mesh_res_interpolation,
                      base_color,
                      model_path, specs_path, model_params_path, latent_codes_path,
                      shaders_dir_path,
                      plot_dim_reduction_paths, mesh_paths,
                      category, split_path, split_dataset_name)

    # Create selection widget children
    dimReductionWidget = DimReductionWidget(app_manager)
    code_img_widget = CodeImgWidget(app_manager = app_manager)
    single_shape_select_widget = MeshWidget(mesh_paths[ShapeType.SELECT_1], app_manager)
    second_shape_select_widget = MeshWidget(mesh_paths[ShapeType.SELECT_2], app_manager)
    
    selection_tab = SelectionWidget(dimReductionWidget, code_img_widget, single_shape_select_widget, second_shape_select_widget, app_manager)
    interpolation_tab = InterpolationWidget(mesh_paths[ShapeType.INTERPOLATE], app_manager)
    menu = MenuWidget(app_manager)

    main_window = MainWindow(selection_tab, interpolation_tab, menu, app_manager)

    sys.exit(app.exec_())