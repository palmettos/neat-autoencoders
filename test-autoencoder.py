import neat
import numpy as np

vec_min = np.array([0. for _ in range(36)])
vec_max = np.array([1. for _ in range(36)])
max_distance = np.linalg.norm(vec_max - vec_min) * 36

inputs = []
for i in range(36):
    l = [0. for _ in range(36)]
    l[i] = 1.
    inputs.append(l)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        encoder, decoder = neat.nn.FeedForwardNetwork.create_autoencoder(genome, config)
        print(encoder)


config = neat.Config(
    neat.AutoencoderGenome, neat.DefaultReproduction,
    neat.DefaultSpeciesSet, neat.DefaultStagnation,
    'autoencoder.cfg'
)


p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(False))
winner = p.run(eval_genomes)
