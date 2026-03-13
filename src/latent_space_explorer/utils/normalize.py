import torch
import numpy as np

class Normalize_parameters():
    def __init__(self, min, range):
        self.min = min
        self.range = range        

def calc_normalize_params(input) -> Normalize_parameters:
    min_values, max_values = np.min(input, axis=0), np.max(input, axis=0)
    range_values = max_values - min_values
    
    return Normalize_parameters(min_values, range_values)

def calc_normalize_params_set(input : dict, CodeSpace) -> dict:
    # Calculate normalize parameters for visualization
    normalize_params = {}
    for space in CodeSpace:
        normalize_params[space] = calc_normalize_params(input[space])

    return normalize_params

def normalize(value, normalize_parameters : Normalize_parameters, element_idx = None) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    
    if element_idx == None:
         return (value - normalize_parameters.min)/normalize_parameters.range
    else:
         return (value - normalize_parameters.min[element_idx])/normalize_parameters.range[element_idx]

def denormalize(fraction, normalize_parameters : Normalize_parameters, element_idx : int = None) -> any:
    if element_idx == None:
         return normalize_parameters.min + fraction * normalize_parameters.range
    else:
         return normalize_parameters.min[element_idx] + fraction * normalize_parameters.range[element_idx]

def normalized_channel_update(input : any, normalize_parameters : Normalize_parameters, channel_idx : int, sign : int, img_value_step) -> np.ndarray:
    element = normalize(input, normalize_parameters, channel_idx)
    element = normalized_update(element, sign, img_value_step)
    return denormalize(element, normalize_parameters, channel_idx)

def normalized_update(value, sign, step) -> np.ndarray:
    return np.clip(value + sign * step, a_min=0, a_max=1)