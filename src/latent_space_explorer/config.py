import os

import torch

from latent_space_explorer.custom_types import DimReductionType, ShapeType


class Config:
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    project_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    print(f"[config] Device is set to {device}")

    def __init__(self):
        self.category = 'cars_vessels_airplanes'

        # Visualization parameters
        self.perplexity = 50
        self.mesh_res_selection = 100
        self.mesh_res_interpolation = 64
        self.base_color = (25, 25, 25)  # 0-255

        self.experiment_path = os.path.join(self.project_path, "experiments", self.category)
        self.shaders_dir_path = os.path.join(self.project_path, 'src/latent_space_explorer', 'shaders')
        # NN paths
        self.model_path = os.path.join(self.project_path, 'DeepSDF')
        self.specs_path = os.path.join(self.experiment_path, 'specs.json')
        self.model_params_path = os.path.join(self.experiment_path, 'ModelParameters/latest.pth')
        self.latent_codes_path = os.path.join(self.experiment_path, 'LatentCodes/latest.pth')
        self.split_path = os.path.join(self.project_path, 'splits', f'{self.category}_train.json')
        self.split_dataset_name = 'ShapeNetV2'

        # Output paths
        self.out_dir = os.path.join(self.experiment_path, 'out')
        self.dim_red_plots_dir = os.path.join(self.out_dir, 'dim_reduction_plots')
        self.reconstructed_meshes_dir = os.path.join(self.out_dir, 'reconstructed_meshes')
        self.plot_dim_reduction_paths = {dim_red_type:
                                         os.path.join(self.dim_red_plots_dir, f'{dim_red_type.name}_plot.png') for dim_red_type in DimReductionType}
        self.mesh_paths = {shape_type:
                           os.path.join(self.reconstructed_meshes_dir, f'{shape_type.name}.ply') for shape_type in ShapeType}
