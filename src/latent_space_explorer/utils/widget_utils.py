import torch
import numpy as np


def fraction_to_graph(plot_ranges, fracts: tuple) -> tuple[float, float]:
    x_fraction, y_fraction = fracts[0], fracts[1]
    dx = plot_ranges['x_range'][1] - plot_ranges['x_range'][0]
    dy = plot_ranges['y_range'][1] - plot_ranges['y_range'][0]
    x_graph = plot_ranges['x_range'][0] + x_fraction * dx
    y_graph = plot_ranges['y_range'][0] + y_fraction * dy
    return x_graph, y_graph


def code_to_img(code) -> np.ndarray:
    img_width = int(code.shape[0]**0.5)
    if torch.is_tensor(code):
        code = code.detach().cpu().numpy()
    img = code.reshape((img_width, img_width)) * 255

    img = np.array(img, dtype='uint8')
    return img
