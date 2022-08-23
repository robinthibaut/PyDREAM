# Copyright (c) 2022. Robin Thibaut, Ghent University

import multiprocessing as mp
import os
import platform
from dataclasses import dataclass
from os.path import dirname, join

import numpy as np

__all__ = ["Machine", "Directories", "HyperParameters", "Dream_shared_vars"]


class Machine(object):
    computer: str = platform.node()


@dataclass
class Directories:
    """Define main directories and file names"""

    # Content directory
    main_dir: str = dirname(os.path.abspath(__file__))
    data_dir: str = join(main_dir, "data")
    results_dir: str = join(main_dir, "outputs")

    package_dir = dirname(main_dir)
    latex_dir = join(package_dir, "report")


@dataclass
class HyperParameters:
    """Define hyperparameters"""

    ...


@dataclass
class Dream_shared_vars:
    history: np.ndarray
    current_positions: np.ndarray
    nchains: int
    cross_probs: np.ndarray
    ncr_updates: np.ndarray
    delta_m: np.ndarray
    gamma_level_probs: np.ndarray
    ngamma_updates: np.ndarray
    delta_m_gamma: np.ndarray
    count: mp.Value
    history_seeded: mp.Value
