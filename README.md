# rrl_binaries_1

This code and supporting Jupyter notebooks document our implentation of the O-C method for
RR Lyrae variables using a modified Hertzsprung method, as well as their Marko-Chaing Monte Carlo
fitting using the `emcee` package. We fit the O-C diagrams with a mixed model containing terms for
both the light-travel time effect, as well as linear period change.

Further details of the techniques are available in the original paper of
[Hajdu et al. (2021)](https://arxiv.org/abs/2105.03750).

These routines were tested on a machine running Linux Mint, using the following versions
of the required packages:
 - `Python` 3.7.10
 - `Numpy` 1.20.1
 - `emcee` 3.0.2
 - `Matplotlib` 3.3.4
 - `Scipy` 1.6.2
 - `scikit-learn` 0.24.1
 - `tqdm` 4.59.0
 - `Numba` 0.53.1
 - `ChainConsumer` 0.31.0
