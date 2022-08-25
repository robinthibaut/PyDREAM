# Copyright (c) 2022. Robin Thibaut, Ghent University

import multiprocessing as mp
import os
import platform
from dataclasses import dataclass
from os.path import dirname, join

import numpy as np

__all__ = ["Machine", "Directories", "HyperParameters", "DreamParameters"]


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
class DreamParameters:
    history: object  # history of the dream
    current_positions: object  # current positions of the dream
    nchains: object  # number of chains
    cross_probs: object  # cross probabilities
    ncr_updates: object  # number of cr updates
    delta_m: object  # delta m
    gamma_level_probs: object  # gamma level probabilities
    ngamma_updates: object  # number of gamma updates
    delta_m_gamma: object  # delta m gamma
    count: mp.Value  # counter for the number of iterations
    history_seeded: mp.Value  # history seeded
