This is a fork of [neat-python](https://github.com/CodeReclaimers/neat-python) with added support for autoencoders with evolvable topologies.

`AutoencoderGenome` creates an encoder and decoder module that can be trained end-to-end.
`FeedForwardNetwork` has a new factory function, `create_autoencoder`, which takes in an `AutoencoderGenome` and returns an `(encoder: DefaultGenome, decoder: DefaultGenome)` tuple. The `AutoencoderGenome` replaces the `num_output` configuration option with `bottleneck_size`.
Example code is provided in `test-autoencoder-objective.py` and `test-autoencoder-novelty.py`.

Current issues:
- There is currently no constraint that guarantees an input node is always connected to a hidden or output node. This doesn't make sense in the context of an autoencoder, so that constraint needs to be added. Currently, the example will only work for fully connected networks with node/connection delete probabilities set to 0, which means only the weights and biases mutate. Adding nodes and connections is optional, though it's not required for the example to maximize the objective function.

Limitations:
- These are simple feedforward networks, so they will only perform well on certain kinds of data -- in particular, data where the features do not have local relationships. In theory, it's still possible for the network to learn these relationships, but probably unreasonably difficult for anything other than toy problems. Use convolutional neural networks for this kind of data.