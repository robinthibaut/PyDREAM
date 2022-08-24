# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:01:34 2015

@author: Erin
"""

# An implementation of example 2 from MT-DREAM(ZS) original Matlab code. (see Laloy and Vrugt 2012)
# 200 dimensional Gaussian distribution

import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from pydream.convergence import Gelman_Rubin
from pydream.core import run_dream
from pydream.parameters import FlatParam
from pydream.config import Directories

main_dir = Directories.main_dir


def Latin_hypercube(minn, maxn, N):
    y = np.random.rand(N, len(minn))
    x = np.zeros((N, len(minn)))

    for mi in range(len(minn)):
        idx = np.random.permutation(N)
        P = (idx - y[:, mi]) / N
        x[:, mi] = minn[mi] + P * (maxn[mi] - minn[mi])

    return x


d = 200
A = 0.5 * np.identity(d) + 0.5 * np.ones((d, d))
C = np.zeros((d, d))
for i in range(d):
    for j in range(d):
        C[i][j] = A[i][j] * np.sqrt((i + 1) * (j + 1))

invC = np.linalg.inv(C)
mu = np.zeros(d)

if d > 150:
    log_F = 0
else:
    log_F = np.log(((2 * np.pi) ** (-d / 2)) * np.linalg.det(C) ** (-1.0 / 2))

# Create initial samples matrix m that will be loaded in as DREAM history file
m = Latin_hypercube(np.linspace(-5, -5, num=d), np.linspace(15, 15, num=d), 1000)

np.save("ndim_gaussian_seed.npy", m)


def likelihood(param_vec):
    logp = log_F - 0.5 * np.sum(param_vec * np.dot(invC, param_vec))

    return logp


starts = [m[chain] for chain in range(3)]

params = FlatParam(test_value=mu)

if __name__ == "__main__":
    niterations = 150000
    # niterations = 150000
    # Run DREAM sampling. Documentation of DREAM options is in Dream.py.
    converged = False
    total_iterations = niterations
    nchains = 3

    sampled_params, log_ps = run_dream(
        [params],
        likelihood,
        niterations=niterations,
        nchains=nchains,
        start=starts,
        start_random=False,
        save_history=True,
        adapt_gamma=False,
        gamma_levels=1,
        tempering=False,
        history_file="ndim_gaussian_seed.npy",
        multitry=5,
        parallel=False,
        model_name="ndim_gaussian",
    )

    for chain in range(len(sampled_params)):
        np.save(
            "ndimgauss_mtdreamzs_3chain_sampled_params_chain_"
            + str(chain)
            + "_"
            + str(total_iterations),
            sampled_params[chain],
        )
        np.save(
            "ndimgauss_mtdreamzs_3chain_logps_chain_"
            + str(chain)
            + "_"
            + str(total_iterations),
            log_ps[chain],
        )

    os.remove("ndim_gaussian_seed.npy")

    # Check convergence and continue sampling if not converged
    GR = Gelman_Rubin(sampled_params)
    print("At iteration: ", total_iterations, " GR = ", GR)
    np.savetxt(
        "ndimgauss_mtdreamzs_3chain_GelmanRubin_iteration_"
        + str(total_iterations)
        + ".txt",
        GR,
    )

    old_samples = sampled_params
    if np.any(GR > 1.2):
        starts = [sampled_params[chain][-1, :] for chain in range(nchains)]
        while not converged:
            total_iterations += niterations

            sampled_params, log_ps = run_dream(
                [params],
                likelihood,
                niterations=niterations,
                nchains=nchains,
                start=starts,
                start_random=False,
                save_history=True,
                adapt_gamma=False,
                gamma_levels=1,
                tempering=False,
                multitry=5,
                parallel=False,
                model_name="ndim_gaussian",
                restart=True,
            )

            for chain in range(len(sampled_params)):
                np.save(
                    "ndimgauss_mtdreamzs_3chain_sampled_params_chain_"
                    + str(chain)
                    + "_"
                    + str(total_iterations),
                    sampled_params[chain],
                )
                np.save(
                    "ndimgauss_mtdreamzs_3chain_logps_chain_"
                    + str(chain)
                    + "_"
                    + str(total_iterations),
                    log_ps[chain],
                )

        old_samples = [
            np.concatenate((old_samples[chain], sampled_params[chain]))
            for chain in range(nchains)
        ]
        GR = Gelman_Rubin(old_samples)
        print("At iteration: ", total_iterations, " GR = ", GR)
        np.savetxt(
            "ndimgauss_mtdreamzs_5chain_GelmanRubin_iteration_"
            + str(total_iterations)
            + ".txt",
            GR,
        )

        if np.all(GR < 1.2):
            converged = True

    # try:
    # plot results
    total_iterations = len(old_samples[0])
    burnin = total_iterations // 2
    samples = np.concatenate(
        (
            old_samples[0][burnin:, :],
            old_samples[1][burnin:, :],
            old_samples[2][burnin:, :],
        )
    )

    ndims = len(old_samples[0][0])
    colors = sns.color_palette(n_colors=ndims)
    for dim in range(ndims):
        fig = plt.figure()
        sns.distplot(samples[:, dim], color=colors[dim])
        fig_name = os.path.join(
            main_dir, f"PyDREAM_example_NDimGauss_dimension_{dim}.png"
        )
        fig.savefig(fig_name, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    # except Exception as e:
    #     print(e)

else:

    run_kwargs = {
        "parameters": [params],
        "likelihood": likelihood,
        "niterations": 150000,
        "nchains": 3,
        "start": starts,
        "start_random": False,
        "save_history": True,
        "adapt_gamma": False,
        "gamma_levels": 1,
        "tempering": False,
        "history_file": "ndim_gaussian_seed.npy",
        "multitry": 5,
        "parallel": False,
        "model_name": "ndim_gaussian",
    }
