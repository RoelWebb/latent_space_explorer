import json
import numpy as np

import matplotlib.pyplot as plt

from latent_space_explorer.config import Config

if Config.device == 'cpu':
    from sklearn.manifold import TSNE, MDS

    def tsne(input_vectors, perplexity):
        return TSNE(n_components=2, perplexity=perplexity).fit_transform(input_vectors)

    def mds(input_vectors):
        return MDS(n_components=2, n_init=1).fit_transform(input_vectors)
else:
    from cuml.manifold import TSNE
    from mdscuda import MDS, minkowski_pairs

    def tsne(input_vectors, perplexity):
        tsne = TSNE(n_components=2, perplexity=perplexity,
                    init='pca', method='fft', learning_rate_method=None, learning_rate=1)
        return tsne.fit_transform(input_vectors)

    def mds(input_vectors):
        DELTA = minkowski_pairs(input_vectors, sqform=False)
        return MDS(n_dims=2, verbosity=0).fit(DELTA, calc_r2=True)

from scipy.spatial import Delaunay

from latent_space_explorer.custom_types import DimReductionType
from latent_space_explorer.utils.profile import Profiler
from latent_space_explorer.utils import widget_utils


def get_barycentric_weights(query_points, tri):
    '''Calculate barycentric weights of triangle corners containing query point'''
    # Find containing triangles and cornerpoint indices
    simplices = tri.find_simplex(query_points)
    simplex_corner_indeces = tri.simplices[simplices]

    # calculate barycentric coordinate for query points
    b = np.array([tri.transform[simplices, :2].dot(np.transpose(query_points - tri.transform[simplices, 2]))])
    barycentric_weights = np.append(b, 1-b.sum())
    return barycentric_weights, simplices, simplex_corner_indeces


def get_category_indeces(json_pathname: str, dataset_name="models_seminar", category_name="augmented_combined"):
    '''Return existing categories and corresponding category per trained object'''
    f = open(json_pathname, "r")
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


def get_reduced_dims(input: np.array, config: Config) -> dict:
    '''Initialize dimension reduction objects and save samples plot as 2D'''
    dim_reduction_dict = {}
    for dim_reduction_type in DimReductionType:
        dim_reduction_dict[dim_reduction_type] = DimReduction2D(input, dim_reduction_type, config)

    return dim_reduction_dict


class DimReduction2D():
    def __init__(self,
                 input_vectors: np.array,
                 dim_reduction_type: DimReductionType,
                 config: Config):

        self.profiler = Profiler(config)

        self.device = config.device

        # Variables for sample labels
        self.samples_json_path = config.split_path
        self.dataset_name = config.split_dataset_name
        self.category = config.category

        # Perform dimensionality reduction
        self.profiler.start_timer(message=f"reduce dim {dim_reduction_type}")
        self.dim_reduction_model = None
        self.reduced_positions = self.fit_and_reduce_dim(dim_reduction_type, input_vectors, config.perplexity)
        self.profiler.report_elapsed_time()
        x_range = [self.reduced_positions[:, 0].min(), self.reduced_positions[:, 0].max()]
        y_range = [self.reduced_positions[:, 1].max(), self.reduced_positions[:, 1].min()]  # y-axis of screen input points up
        self.dim_reduced_ranges = {'x_range': x_range,
                                   'y_range': y_range}
        print(f"Performed {dim_reduction_type} dimensionality reduction")

        # Peform Delaunay for later barycentric weighting and plotting
        self.tri = Delaunay(self.reduced_positions)
        print("Performed delaunay triangulation")

        self.save_plot_reduced_dim(config.plot_dim_reduction_paths[dim_reduction_type], config.base_color)
        print("Saved plot")

    def fit_and_reduce_dim(self, dim_reduction_type: DimReductionType, input_vectors: np.array, perplexity: int) -> any:
        if dim_reduction_type == DimReductionType.tSNE:
            return tsne(input_vectors, perplexity)
        elif dim_reduction_type == DimReductionType.MDS:
            return mds(input_vectors)

    def save_plot_reduced_dim(self, output_path: str, base_color: tuple) -> None:
        fig, ax = plt.subplots()
        categories, index_to_category_index = get_category_indeces(self.samples_json_path,
                                                                   self.dataset_name,
                                                                   self.category)

        scatter = ax.scatter(self.reduced_positions[:, 0], self.reduced_positions[:, 1], c=index_to_category_index, s=40)

        handles, _ = scatter.legend_elements()
        legend = ax.legend(handles, categories)
        ax.add_artist(legend)

        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, facecolor=[channel/255.0 for channel in base_color],
                    dpi=150)

    def get_category_indeces(self) -> tuple[list, list]:
        '''Return existing categories and corresponding category per trained object'''
        f = open(self.samples_json_path, "r")
        shapes_dict = json.loads(f.read())
        names = shapes_dict[self.dataset_name][self.category]

        categories = []
        index_lookup = {}
        index_to_category_index = []
        for name in names:
            category = name.split("_")[0]
            # found new category, add to existing categories and new lookup
            if category not in categories:
                index_lookup[category] = len(categories)
                categories.append(category)
            index_to_category_index.append(index_lookup[category])

        return categories, index_to_category_index

    def calc_Delaunay_idx_and_weights(self, query_points: np.array) -> tuple[np.array, np.array]:
        '''Calculate Delaunay triangle corner indices and barycentric weights containing query point'''
        self.profiler.start_timer(message="query simplex corner indices and barycentric weights")
        containing_simplices = self.tri.find_simplex(query_points)
        simplex_corner_indeces = self.tri.simplices[containing_simplices]

        # calculate barycentric weights
        b = np.array([self.tri.transform[containing_simplices, :2].dot(np.transpose(query_points - self.tri.transform[containing_simplices, 2]))])
        barycentric_weights = np.append(b, 1-b.sum())
        self.profiler.report_elapsed_time()
        return simplex_corner_indeces, barycentric_weights

    def fract_to_barycentric_weights(self, plot_pos_fracts: list) -> tuple[np.array, np.array]:
        pos_reduced = widget_utils.fraction_to_graph(self.dim_reduced_ranges, plot_pos_fracts)
        simplex_corner_indeces, barycentric_weights = self.calc_Delaunay_idx_and_weights(pos_reduced)
        return simplex_corner_indeces, barycentric_weights
