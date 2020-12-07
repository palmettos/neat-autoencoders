This is a fork of [neat-python](https://github.com/CodeReclaimers/neat-python) with added support for autoencoders with evolvable topologies.

`AutoencoderGenome` creates an encoder and decoder module that can be trained end-to-end.
`FeedForwardNetwork` has a new factory function, `create_autoencoder`, which takes in an `AutoencoderGenome` and returns an `(encoder: FeedForwardNetwork, decoder: FeedForwardNetwork)` tuple. The `AutoencoderGenome` replaces the `num_output` configuration option with `bottleneck_size`.
Example code is provided in `test-autoencoder-objective.py` and `test-autoencoder-novelty.py`.

For the objective example, once the network has reached a satisfactory fitness, a plot will be shown with sliders that allow you to modify the bottleneck inputs to the decoder module.

Why?
- Because I think it's cool -- but honestly, I'm not sure. I imagined this potentially opening up some routes of experimentation with all the beautiful things available in neuroevolution, like novelty search and multi-objective optimization.

Limitations:
- These are simple feedforward networks, so they will only perform well on certain kinds of data -- in particular, data where the features do not have local relationships. In theory, it's still possible for the network to learn these relationships, but probably unreasonably difficult for anything other than toy problems. Use convolutional neural networks for this kind of data.
