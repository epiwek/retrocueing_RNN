# retrocueing_RNN

[![DOI](https://zenodo.org/badge/531965284.svg)](https://zenodo.org/badge/latestdoi/531965284)

This repo contains simulations published in the following paper:

Piwek, Stokes, & Summerfield (2023) A recurrent neural network model of prefrontal brain activity during a working memory task. PLOS Computational Biology (in press)

This project trains a simple RNN model to perform different variants of a delayed recall (retro-cueing) task. The network receives two colour-location stimuli and needs to maintain them across a memory delay. It subsequently receives a 'cue' input, providing information about which of the two stimuli it should report after a second memory delay. More specifically, the cue provides information about the *location* of the required stimulus, and the network is asked to report its *colour* (which it needs to retrieve from its internal dynamics).

After training, we analyse the representations formed by the recurrent layer of the network, to understand how it solves the task. We focus on how the colour-location information is transformed across the different task stages. We also analyse the network training dynamics and connectivity, to gain deeper insight into the task solution learnt by the RNN model.

We find that the internal representations learnt by the RNN are a close match to those previously observed in macaque brains (in the prefrontal cortex; see Panichello and Buschman 2021). At the beginning of the trial episode, the network maintains the information about the two stimulus items in orthogonal subspaces. Following the retro-cue input, this information is rotated in the neuronal space, such that the Cued items are placed on parallel manifolds, allowing a common colour readout mechanism to be used irrespective of the retro-cue location.

We extend these results by analysing variants of the original task. This includes a manipulation of the relative lengths of the memory delays. We also analyse behavioural performance and internal representations on a probabilistic condition where the retro-cue input is not fully reliable (i.e., it can point to the incorrect item).

To run the simulations, pick an appropriate experimental configuration file from the 'constants/' folder and pass it to the run_experiment() function in 'main.py'.
