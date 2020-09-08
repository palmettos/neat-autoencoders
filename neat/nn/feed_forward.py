from neat.graphs import feed_forward_layers


class FeedForwardNetwork(object):
    def __init__(self, inputs, outputs, node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.values = dict((key, 0.0) for key in inputs + outputs)

    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        for k, v in zip(self.input_nodes, inputs):
            self.values[k] = v

        for node, act_func, agg_func, bias, response, links in self.node_evals:
            node_inputs = []
            for i, w in links:
                node_inputs.append(self.values[i] * w)
            s = agg_func(node_inputs)
            self.values[node] = act_func(bias + response * s)

        return [self.values[i] for i in self.output_nodes]

    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (a FeedForwardNetwork). """

        # Gather expressed connections.
        connections = [cg.key for cg in genome.connections.values() if cg.enabled]

        layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections)
        node_evals = []
        for layer in layers:
            for node in layer:
                inputs = []
                node_expr = [] # currently unused
                for conn_key in connections:
                    inode, onode = conn_key
                    if onode == node:
                        cg = genome.connections[conn_key]
                        inputs.append((inode, cg.weight))
                        node_expr.append("v[{}] * {:.7e}".format(inode, cg.weight))

                ng = genome.nodes[node]
                aggregation_function = config.genome_config.aggregation_function_defs.get(ng.aggregation)
                activation_function = config.genome_config.activation_defs.get(ng.activation)
                node_evals.append((node, activation_function, aggregation_function, ng.bias, ng.response, inputs))

        return FeedForwardNetwork(config.genome_config.input_keys, config.genome_config.output_keys, node_evals)

    @staticmethod
    def create_autoencoder(genome, config):
        encoder_connections = [cg.key for cg in genome.encoder.connections.values() if cg.enabled]
        decoder_connections = [cg.key for cg in genome.decoder.connections.values() if cg.enabled]

        encoder_layers = feed_forward_layers(
            config.genome_config.encoder_input_keys,
            config.genome_config.encoder_output_keys,
            encoder_connections
        )
        encoder_node_evals = []
        for layer in encoder_layers:
            for node in layer:
                inputs = []
                node_expr = [] # currently unused
                for conn_key in encoder_connections:
                    inode, onode = conn_key
                    if onode == node:
                        cg = genome.encoder.connections[conn_key]
                        inputs.append((inode, cg.weight))
                        node_expr.append("v[{}] * {:.7e}".format(inode, cg.weight))

                ng = genome.encoder.nodes[node]
                aggregation_function = config.genome_config.aggregation_function_defs.get(ng.aggregation)
                activation_function = config.genome_config.activation_defs.get(ng.activation)
                encoder_node_evals.append((node, activation_function, aggregation_function, ng.bias, ng.response, inputs))

        decoder_layers = feed_forward_layers(
            config.genome_config.decoder_input_keys,
            config.genome_config.decoder_output_keys,
            decoder_connections
        )
        decoder_node_evals = []
        for layer in decoder_layers:
            for node in layer:
                inputs = []
                node_expr = [] # currently unused
                for conn_key in decoder_connections:
                    inode, onode = conn_key
                    if onode == node:
                        cg = genome.decoder.connections[conn_key]
                        inputs.append((inode, cg.weight))
                        node_expr.append("v[{}] * {:.7e}".format(inode, cg.weight))

                ng = genome.decoder.nodes[node]
                aggregation_function = config.genome_config.aggregation_function_defs.get(ng.aggregation)
                activation_function = config.genome_config.activation_defs.get(ng.activation)
                decoder_node_evals.append((node, activation_function, aggregation_function, ng.bias, ng.response, inputs))

        encoder = FeedForwardNetwork(
            config.genome_config.encoder_input_keys,
            config.genome_config.encoder_output_keys,
            encoder_node_evals
        )

        decoder = FeedForwardNetwork(
            config.genome_config.decoder_input_keys,
            config.genome_config.decoder_output_keys,
            decoder_node_evals
        )

        return encoder, decoder