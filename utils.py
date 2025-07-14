import torch
import random
import numpy as np
from torch import nn
from typing import Tuple
from plyfile import PlyData, PlyElement


def read_ply(path):
    with open(path, "rb") as f:
        plydata = PlyData.read(f)
        x = np.array(plydata["vertex"]["x"])
        y = np.array(plydata["vertex"]["y"])
        z = np.array(plydata["vertex"]["z"])
        vertex = np.stack([x, y, z], axis=1)
    return vertex


def write_ply(points, filename, text=False):
    """input: Nx3, write points to filename as PLY format."""
    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    el = PlyElement.describe(vertex, "vertex", comments=["vertices"])
    with open(filename, mode="wb") as f:
        PlyData([el], text=text).write(f)


def set_randomness(random_seed: int = 42):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def count_model_parameters(model: nn.Module) -> Tuple[int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def read_text(path):
    with open(path, "r") as f:
        res = f.readlines()

    return [line.strip() for line in res]
