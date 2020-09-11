import neat
import numpy as np
from sklearn.neighbors import KDTree


dimensions = 6
inputs = [
    [1., 1., 1., 0., 0., 0.],
    [0., 1., 1., 1., 0., 0.],
    [0., 0., 1., 1., 1., 0.],
    [0., 0., 0., 1., 1., 1.],
    [1., 0., 0., 0., 1., 1.],
    [1., 1., 0., 0., 0., 1.]
]

max_accuracy = dimensions ** 2

def eval_genomes(genomes, config):
    global current_best_accuracy
    for genome_id, genome in genomes:
        accuracy = max_accuracy
        encoder, decoder = neat.nn.FeedForwardNetwork.create_autoencoder(genome, config)
        for input in inputs:
            bottleneck_output = encoder.activate(input)
            reconstructed = decoder.activate(bottleneck_output)
            for expected, output in zip(input, reconstructed):
                accuracy -= (expected - output) ** 2
        genome.fitness = accuracy


config = neat.Config(
    neat.AutoencoderGenome, neat.DefaultReproduction,
    neat.DefaultSpeciesSet, neat.DefaultStagnation,
    'test-autoencoder-objective.cfg'
)


p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(False))
winner = p.run(eval_genomes)
encoder, decoder = neat.nn.FeedForwardNetwork.create_autoencoder(winner, config)
for input in inputs:
    print(f'input: {input}')
    bottleneck_output = encoder.activate(input)
    reconstructed = decoder.activate(bottleneck_output)
    print(f'output: {reconstructed}')