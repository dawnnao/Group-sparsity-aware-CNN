# Group-sparsity-aware-CNN

![Tensorflow Badge](https://aleen42.github.io/badges/src/tensorflow.svg)

This repository is the implementation of group sparsity-aware CNN for
continuous missing data recovery of structural health monitoring as described in the paper

> ["Group sparsity-aware convolutional neural network for continuous missing data recovery of structural health monitoring"
by Zhiyi Tang, Yuequan Bao, and Hui Li.](https://journals.sagepub.com/doi/full/10.1177/1475921720931745)

Packages dependencies are listed in `requirements.txt`. The GS-aware CNN, and the data pre- and post-processing are packed into
the main function `gsn()`. It works as an automatic work flow once given required parameters and the data-to-recover.
Two example data sets have been included in folder `\simulation_El_Centro` and `\simulation_impulse`. Run `GS-aware_CNN.py` with the
default parameters to check the examples.