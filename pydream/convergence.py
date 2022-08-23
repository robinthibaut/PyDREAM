import numpy as np


def Gelman_Rubin(sampled_parameters):
    """
    Calculate the Gelman-Rubin diagnostic for a set of MCMC samples.
    Parameters
    ----------
    sampled_parameters: list
        A list of arrays of MCMC samples for each parameter.

    Returns
    -------
    R: array
        The Gelman-Rubin diagnostic.
    """
    nsamples = len(sampled_parameters[0])
    nchains = len(sampled_parameters)
    nburnin = nsamples // 2

    chain_var = [
        np.var(sampled_parameters[chain][nburnin:, :], axis=0)
        for chain in range(nchains)
    ]

    W = np.mean(chain_var, axis=0)

    chain_means = [
        np.mean(sampled_parameters[chain][nburnin:, :], axis=0)
        for chain in range(nchains)
    ]

    B = np.var(chain_means, axis=0)

    var_est = (W * (1 - (1.0 / nsamples))) + B

    Rhat = np.sqrt(np.divide(var_est, W))

    return Rhat
