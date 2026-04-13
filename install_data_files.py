import os
import gdown

project_root = os.path.dirname(__file__)
experiments_path = os.path.join(project_root, "experiments")
cars_vessels_airplanes_path = os.path.join(experiments_path, "cars_vessels_airplanes")
splits_path = os.path.join(project_root, "splits", "cars_vessels_airplanes_train.json")

os.makedirs(cars_vessels_airplanes_path, exist_ok=True)

gdown.download_folder(id="1mA9hjZMR13XHGhD1bAVGGHNwCzdWQf53", output=cars_vessels_airplanes_path, quiet=False)
gdown.download(id="15MFd-__SyXKUjmoOrmPO_WaOPSs5liL0", output=splits_path, quiet=False)
