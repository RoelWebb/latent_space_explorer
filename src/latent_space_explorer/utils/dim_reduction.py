import json
import numpy as np

import matplotlib.pyplot as plt

from tsnecuda import TSNE
from mdscuda import MDS, minkowski_pairs
from scipy.spatial import Delaunay

from latent_space_explorer.custom_types import DimReductionType
from latent_space_explorer.utils.profile import Profiler
from latent_space_explorer.utils import widget_utils

def get_barycentric_weights(query_points, tri):
        '''Calculate barycentric weights of triangle corners containing query point'''
        # Find containing triangles and cornerpoint indices
        containing_simplices = tri.find_simplex(query_points)
        simplex_corner_indeces = tri.simplices[containing_simplices]        
        
        # calculate barycentric coordinate for query points
        b = np.array([tri.transform[containing_simplices,:2].dot(np.transpose(query_points - tri.transform[containing_simplices,2]))])
        barycentric_weights = np.append(b, 1-b.sum())
        return barycentric_weights, containing_simplices, simplex_corner_indeces


def get_category_indeces(json_pathname, dataset_name="models_seminar", category_name="augmented_combined"):
    '''Return existing categories and corresponding category per trained object'''
    f = open (json_pathname, "r")
    shapes_dict = json.loads(f.read())
    names = shapes_dict[dataset_name][category_name]

    categories = []
    index_lookup = {}
    index_to_category_index = []
    for i, name in enumerate(names):
        category = name.split("_")[0]
        # found new category, add to existing categories and new lookup
        if category not in categories:
            index_lookup[category] = len(categories)
            categories.append(category)
        index_to_category_index.append(index_lookup[category])

    return categories, index_to_category_index
    
def reduced_pos_to_latent(reduced_pos, tri):
    barycentric_weights, containing_simplices, simplex_corner_indeces = get_barycentric_weights(reduced_pos, tri)
    return simplex_corner_indeces, barycentric_weights

def get_reduced_dims(input : np.array,
                     dim_reduction_plot_paths : dict, perplexity : int,
                     category_name : str, objects_json_path : str, dataset_name : str,
                     base_color) -> dict:
        '''Initialize dimension reduction objects and save samples plot as 2D'''
        dim_reduction_dict = {}
        for dim_reduction_type in DimReductionType:
            dim_reduction_dict[dim_reduction_type] = DimReduction2D(input,
                                                                      dim_reduction_type,
                                                                      dim_reduction_plot_paths[dim_reduction_type], perplexity,
                                                                      samples_json_path = objects_json_path, dataset_name = dataset_name, category_name = category_name,
                                                                      base_color=base_color)

        return dim_reduction_dict

class DimReduction2D():
    def __init__(self,
                 input_vectors,
                 dim_reduction_type,
                 plot_path, perplexity=None,
                 samples_json_path = None, dataset_name = None, category_name = None,
                 base_color=None):
        
        self.profiler = Profiler()

        # Variables for sample labels
        self.samples_json_path = samples_json_path
        self.dataset_name = dataset_name
        self.category_name = category_name

        # Perform dimensionality reduction
        self.profiler.start_timer(message=f"reduce dim {dim_reduction_type}")
        self.dim_reduction_model = None
        self.reduced_positions = self.fit_and_reduce_dim(dim_reduction_type, input_vectors, perplexity)
        self.profiler.report_elapsed_time()
        x_range = [self.reduced_positions[:,0].min(), self.reduced_positions[:,0].max()]
        y_range = [self.reduced_positions[:,1].max(), self.reduced_positions[:,1].min()] # OpenGl screen input has the y-axis pointing upwards
        self.dim_reduced_ranges = {'x_range' : x_range,
                       'y_range' : y_range}
        print(f"Performed {dim_reduction_type} dimensionality reduction")

        # Peform Delaunay for later barycentric weighting and plotting
        self.tri = Delaunay(self.reduced_positions)
        print("Performed delaunay triangulation")

        self.save_plot_reduced_dim(plot_path, base_color)
        print("Saved plot")

    def fit_and_reduce_dim(self, dim_reduction_type : DimReductionType, input_vectors : np.array, perplexity : int) -> any:
        if dim_reduction_type == DimReductionType.tSNE:
            return TSNE(n_components=2, perplexity=perplexity).fit_transform(input_vectors)
        elif dim_reduction_type == DimReductionType.MDS:
            DELTA = minkowski_pairs(input_vectors, sqform = False)  # this returns a matrix of pairwise distances in longform
            return MDS(n_dims = 2, verbosity = 0).fit(DELTA, calc_r2=True)

    def save_plot_reduced_dim(self, output_path : str, base_color : tuple) -> None:
        fig, ax = plt.subplots()
        categories, index_to_category_index = get_category_indeces(self.samples_json_path, dataset_name=self.dataset_name, category_name=self.category_name)
        
        scatter = ax.scatter(self.reduced_positions[:, 0], self.reduced_positions[:, 1], c=index_to_category_index, s=40)
        
        handles, _ = scatter.legend_elements()
        legend = ax.legend(handles, categories)
        ax.add_artist(legend)

        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, facecolor=[channel/255.0 for channel in base_color], dpi=150)

    def get_category_indeces(json_pathname : str, data_dir_path : str = "models_seminar", category_name : str = "augmented_combined") -> tuple[list, list]:
        '''Return existing categories and corresponding category per trained object'''
        f = open (json_pathname, "r")
        shapes_dict = json.loads(f.read())
        names = shapes_dict[data_dir_path][category_name]

        categories = []
        index_lookup = {}
        index_to_category_index = []
        for i, name in enumerate(names):
            category = name.split("_")[0]
            # found new category, add to existing categories and new lookup
            if category not in categories:
                index_lookup[category] = len(categories)
                categories.append(category)
            index_to_category_index.append(index_lookup[category])

        return categories, index_to_category_index

    def calc_Delaunay_idx_and_weights(self, query_points : np.array) -> tuple[np.array, np.array]:
        '''Calculate Delaunay triangle corner indices and barycentric weights containing query point'''
        self.profiler.start_timer(message="query simplex corner indices and barycentric weights")
        containing_simplices = self.tri.find_simplex(query_points)
        simplex_corner_indeces = self.tri.simplices[containing_simplices]        

        # calculate barycentric weights
        b = np.array([self.tri.transform[containing_simplices,:2].dot(np.transpose(query_points - self.tri.transform[containing_simplices,2]))])
        barycentric_weights = np.append(b, 1-b.sum())
        self.profiler.report_elapsed_time()
        return simplex_corner_indeces, barycentric_weights
    
    def fract_to_barycentric_weights(self, plot_pos_fracts : list) -> tuple[np.array, np.array]:
        pos_reduced = widget_utils.fraction_to_graph(self.dim_reduced_ranges, plot_pos_fracts)
        simplex_corner_indeces, barycentric_weights = self.calc_Delaunay_idx_and_weights(pos_reduced)
        return simplex_corner_indeces, barycentric_weights