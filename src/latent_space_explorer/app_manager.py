import numpy as np
import torch

from PyQt5.QtCore import QObject, pyqtSignal

from torch_pca import PCA

from latent_space_explorer.config import Config
from latent_space_explorer.custom_types import CodeSpace, VisualizeShapeMode, ShapeType
from latent_space_explorer.utils.profile import Profiler
from latent_space_explorer.utils.IO import load_latent_codes
from latent_space_explorer.utils.dim_reduction import DimReductionType, get_reduced_dims
from latent_space_explorer.utils.widget_utils import code_to_img
from latent_space_explorer.utils import normalize
from latent_space_explorer.utils import mesh_VARIATION as mesh # mesh_VARIATION.py is a very slighlty adjusted version of the mesh.py script in the DeepSDF paper implementation written by Park et al.


class AppManager(QObject):
    """Handles communication between widgets and performs required processing"""
    # Connection between app manager and UI widgets
    interpolation_tab_activated = pyqtSignal()
    dim_reduction_changed = pyqtSignal(str)
    visualize_mode_changed = pyqtSignal(int)
    rotation_speed_changed = pyqtSignal(int)
    code_img_changed = pyqtSignal(np.ndarray)
    
    def __init__(self, config : Config):
        super().__init__()
        
        # TODO: look at existing profiler packages
        self.profiler = Profiler(config)

        self.device = config.device # Only for the network and its input

        # Visualization parameters
        self.visualize_mode = VisualizeShapeMode.SINGLE
        self.code_space = CodeSpace.PCA
        self.dim_reduction_type = DimReductionType.tSNE
        self.perplexity = config.perplexity
        self.mesh_res_selection = config.mesh_res_selection
        self.mesh_res_interpolation = config.mesh_res_interpolation
        self.img_value_step = 0.1 # Fraction steps for adjusting latent codes on latent img selection
        self.rotations_per_minute = 8
        self.base_color = config.base_color

        # Set paths
        self.latent_codes_path = config.latent_codes_path
        self.shaders_dir_path = config.shaders_dir_path
        self.plot_paths = config.plot_dim_reduction_paths
        self.mesh_paths = config.mesh_paths

        # Load all latent codes for later distance weighting
        self.latent_codes = load_latent_codes(self.latent_codes_path).detach().cpu().numpy()
        
        # Initialize shape_reconstructor and laten code input
        self.shape_reconstructor = mesh.Shape_reconstructor(config)
        self.active_shape_codes = torch.zeros((3, self.latent_codes.shape[1]), device=self.device)
        
        # Perform dimensionality reduction and save plots
        self.dim_reductions = get_reduced_dims(self.latent_codes, config=config)

        # Fit PCA model to latent codes
        self.pca_model = PCA()
        self.profiler.start_timer(message = "Fitted PCA model to latent codes")
        PCA_codes = self.pca_model.fit_transform(torch.from_numpy(self.latent_codes).to(device=self.device)).detach().cpu().numpy()
        self.profiler.report_elapsed_time()

        # Calculate normalizing parameter for latent code visualization and normalized channel adjustment
        self.code_normalize_params = normalize.calc_normalize_params_set({CodeSpace.LATENT : self.latent_codes,
                                                                          CodeSpace.PCA : PCA_codes}, CodeSpace)
        
    # Called from menu widget through pyqtSignal
    def set_selection_resolution(self, resolution : int) -> None:
        self.mesh_res_selection = resolution
    
    def set_interpolation_resolution(self, resolution : int) -> None:
        self.mesh_res_interpolation = resolution

    def reduced_dim_click(self, x : float, y : float, button : int) -> None:
        """Update barycentric weighted latent code and reconstruct shape mesh, when clicking reduced dimensionality image
        ### Input:
        - x, y: normalized/fraction position on clicked reduced dimensionality image
        - button: 0 = left click, 1 = right click (for single shape visualization and double shape respectively)
        """
        print(f"Clicked dimensionality reduction image at x:{x}, y:{y}, with button {button}")

        plot_pos_fracts = (x,y)

        button-=1
        if button == 0 or button == 1: # Clicked left or right
            simplex_corner_indeces, barycentric_weights = self.dim_reductions[self.dim_reduction_type].fract_to_barycentric_weights(plot_pos_fracts)
            if np.all(barycentric_weights >= 0) and np.all(barycentric_weights <= 1): # Avoid extrapolation
                if VisualizeShapeMode(button) is not self.visualize_mode:
                    self.visualize_mode = VisualizeShapeMode(button)
                    self.visualize_mode_changed.emit(self.visualize_mode.value)

                self.update_barycentric_weighted_codes(button, barycentric_weights, simplex_corner_indeces)

                # Visualize and reconstruct
                self.set_code_img()
                self.reconstruct_from_dim_reduction()
            else:
                print(f"No code implemented for extrapolation (unable to peform barycentric weighting at selected location)")

    def code_click(self, element_idx : int, button : int) -> None:
        """Update single channel of latent code and reconstruct shape mesh, when clicking latent image texel/channel"""
        self.visualize_mode = VisualizeShapeMode.SINGLE
        self.visualize_mode_changed.emit(self.visualize_mode.value)

        # Left button increase channel, right decrease
        if button == 1:
            sign = 1.0
        elif button == 2:
            sign = -1.0

        self.normalized_code_channel_update(element_idx, sign)

    def linear_interpolate(self, interpolate_percentage : int) -> None:
        """Update linear interpolated latent code and reconstruct shape, connected with slider update of interpolation widget"""
        fract = interpolate_percentage / 100.0
        self.active_shape_codes[2] = (1.0 - fract) * self.active_shape_codes[0] + fract * self.active_shape_codes[1]
        self.shape_reconstructor.reconstruct_shape(self.active_shape_codes[2], self.mesh_paths[ShapeType.INTERPOLATE], mesh_resolution = self.mesh_res_interpolation)

    def update_barycentric_weighted_codes(self, shape_idx : int, barycentric_weights : np.array, simplex_corner_indeces : np.array) -> None:
        """Update active weighted latent code and PCA code"""
        self.active_shape_codes[shape_idx] = torch.from_numpy(np.sum(barycentric_weights.reshape(-1, 1) * self.latent_codes[simplex_corner_indeces], axis=0)).to(self.device)
    
    def normalized_code_channel_update(self, element_idx : int, sign : int) -> None:
        """Update a single channel of the latent code in selected space after normalization and storing updated (normalized) image"""
        
        # Convert to PCA for normalized update if required
        if self.code_space is CodeSpace.PCA:
            self.active_shape_codes[0] = self.pca_model.transform(self.active_shape_codes[0].unsqueeze(0)).squeeze()

        # Normalize, perform normalized update and denormalize
        self.active_shape_codes[0, element_idx] = normalize.normalized_channel_update(self.active_shape_codes[0, element_idx], self.code_normalize_params[self.code_space], element_idx, sign, self.img_value_step)
        
        # Convert back to latent space if required
        if self.code_space is CodeSpace.PCA:
            self.active_shape_codes[0] = self.pca_model.inverse_transform(self.active_shape_codes[0])

        # Set, visualize code and reconstruct shape
        self.set_code_img()
        self.reconstruct_from_dim_reduction()

    def set_code_img(self) -> None:
        """Update the image of connected code image widget. When in single shape selection mode, set image to normalized latent code in the correct space or for double shape selection set their difference"""
        
        # Transform latent code to correct space if required
        if self.code_space is CodeSpace.PCA:
            self.active_shape_codes = self.pca_model.transform(self.active_shape_codes)

        # Visualize single code or difference between shape 1 and 2
        if self.visualize_mode is VisualizeShapeMode.SINGLE:
            self.code_img_changed.emit(code_to_img(normalize.normalize(self.active_shape_codes[0], normalize_parameters = self.code_normalize_params[self.code_space])))
        elif self.visualize_mode is VisualizeShapeMode.DOUBLE:
            self.code_img_changed.emit(code_to_img(np.abs(normalize.normalize(self.active_shape_codes[0], normalize_parameters = self.code_normalize_params[self.code_space])
                                                          - normalize.normalize(self.active_shape_codes[1], normalize_parameters = self.code_normalize_params[self.code_space]))))

        # Convert back to latent space if required
        if self.code_space is CodeSpace.PCA:
            self.active_shape_codes = self.pca_model.inverse_transform(self.active_shape_codes)
        
    def reconstruct_from_dim_reduction(self) -> None:
        """Update the latent code visualization and reconstruct selection shape 1 or 2 (depending on last click/selection mode)"""
        self.shape_reconstructor.reconstruct_shape(self.active_shape_codes[self.visualize_mode.value], self.mesh_paths[ShapeType(self.visualize_mode.value)], mesh_resolution = self.mesh_res_selection)

    def update_code_space(self, i : int) -> None:
        self.code_space = CodeSpace(i)
        self.normalized_code_channel_update(element_idx=0, sign=0) # Convert image to correct space        
        print(f"Code space changed to {self.code_space}")
    
    def update_dim_reduction_type(self, i : int) -> None:
        self.dim_reduction_type = DimReductionType(i)
        self.dim_reduction_changed.emit(self.plot_paths[self.dim_reduction_type])
        print(f"Dim reduction changed to {self.dim_reduction_type}")

    def changed_tab(self) -> None:
        self.linear_interpolate(0)
        self.interpolation_tab_activated.emit()

    def update_rotation_speed(self, new_rotation_per_minute : int) -> None:
        """Update app_manager rotation speed atribute and call rotation speed update function of mesh widgets through Qt signal"""
        self.rotations_per_minute = new_rotation_per_minute
        self.rotation_speed_changed.emit(new_rotation_per_minute)