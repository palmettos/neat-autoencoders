This is a fork of [neat-python](https://github.com/CodeReclaimers/neat-python) with added support for autoencoders with evolvable topologies.

`AutoencoderGenome` creates an encoder and decoder module that can be trained end-to-end.
`FeedForwardNetwork` has a new factory function, `create_autoencoder`, which takes in an `AutoencoderGenome` and returns an `(encoder: FeedForwardNetwork, decoder: FeedForwardNetwork)` tuple. The `AutoencoderGenome` replaces the `num_output` configuration option with `bottleneck_size`.
Example code is provided in `test-autoencoder-objective.py` and `test-autoencoder-novelty.py`.

Current issues:
- There is currently no constraint that guarantees an input node is always connected to a hidden or output node. In the context of autoencoders, disconnected inputs are usually problematic. In some cases it doesn't matter, like if input 1 tells you everything you need to know about input 2, then input is redundant. In most cases you probably want all input nodes to have at least one connection to a hidden or output node.

Limitations:
- These are simple feedforward networks, so they will only perform well on certain kinds of data -- in particular, data where the features do not have local relationships. In theory, it's still possible for the network to learn these relationships, but probably unreasonably difficult for anything other than toy problems. Use convolutional neural networks for this kind of data.