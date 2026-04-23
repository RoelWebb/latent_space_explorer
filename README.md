![Demo](application_screen_record.gif)

# Latent code explorer
This application enables the exploration of a learned latent space of the [DeepSDF](https://github.com/facebookresearch/DeepSDF) model proposed by *Park et al.*. For the demo, a smaller model of the proposed *DeepSDF* architecture was trained on a subset of 3 categories (vessels, cars and airplanes) from the [ShapeNetCore](https://huggingface.co/datasets/ShapeNet/ShapeNetCore) (v2) dataset, with roughly 200 shapes each.

To intuitively visualize the learned 64 dimensional latent code per shape, they are projected to a reduced dimensionality of 2. Selecting an arbitrary point in this 2D space, barycentric weights are calculated to calculate a weighted latent code, which is reconstructed to a new 3D shape. Finally, 2 latent codes can be selected and interpolated.

## Install

```bash
# Clone latent space explorer repo and DeepSDF model
git clone ----recursive https://github.com/RoelWebb/latent_space_explorer 
cd latent_space_explorer
```

```bash
# Setup venv environment, install dependencies and activate environment
uv sync
source .venv/bin/activate
```


<details>
<summary>pip install dependencies (alternative to uv)</summary>

```bash
# Setup venv environment and install dependencies
python3.10 -m venv .venv
source .venv/bin/activate

pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install tsnecuda==3.0.1+cu118 -f  https://tsnecuda.isx.ai/tsnecuda_stable.html
pip install -e .
```
</details>

```bash
# Download pretrained model weights, specs, split for category annotations and pretrained shape latent codes
python install_data_files.py
```


## Run the app
Run the following command from the project directory root:
```bash
python -m src.latent_space_explorer.app
```

The application consists of a tab for code selection and shape interpolation, both with a menu to adjust relevant parameters (shape resolution is only updated after a new code selection). The code selection tab contains the reduced dimensionality map (top-left) to select a new single latent code with a left-click and an additional latent code with right-click to enable shape interpolation in the interpolation tab by adjusting the slider. The shape code visualization (bottom-left) enables increasing or decreasing of a selected latent code element with left- or right-click respectively.

## License
Latent space explorer is released under the MIT License for its permissive open-source use and distribution, as stated in the [LICENSE file](LICENSE).

## Acknowledgement
The script [mesh_VARIATION.py](src/latent_space_explorer/utils/mesh_VARIATION.py) is a direct copy of [`deep_sdf/mesh.py`](https://github.com/facebookresearch/DeepSDF/blob/main/deep_sdf/mesh.py) from [DeepSDF](https://github.com/facebookresearch/DeepSDF) with very minor adjustments.

## Future todos
- [ ] Profile code for performance bottlenecks (try decreasing points where DeepSDF is infered and check repeated self.vao_mesh initialization in [mesh.py](src/latent_space_explorer/UI/mesh.py))
- [ ] Check correct stretching/sizing of reduced dimensionality plot for click position and draw reduced dimensionality with PyQt for higher resolution (clicking outside scatter points should not update the mesh)
- [ ] Add more pretrained neural networks such as Spaghetti
- [ ] Train model on more points per shape and more shapes for higher shape quality
