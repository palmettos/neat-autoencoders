import neat
import numpy as np
from sklearn.neighbors import KDTree

# the number of neighbors to consider when calculating novelty
k = 2
# the distance a behavior must exceed for archival
archive_threshold = 0.01

dimensions = 6
inputs = [
    [1., 1., 1., 0., 0., 0.],
    [0., 1., 1., 1., 0., 0.],
    [0., 0., 1., 1., 1., 0.],
    [0., 0., 0., 1., 1., 1.],
    [1., 0., 0., 0., 1., 1.],
    [1., 1., 0., 0., 0., 1.]
]

all_behaviors = []
max_accuracy = dimensions ** 2
current_best_accuracy = 0.

def eval_genomes(genomes, config):
    global current_best_accuracy
    # items: (genome: AutoencoderGenome, accuracy: int)[]
    gen_behaviors = {}
    for genome_id, genome in genomes:
        accuracy = max_accuracy
        encoder, decoder = neat.nn.FeedForwardNetwork.create_autoencoder(genome, config)
        for input in inputs:
            bottleneck_output = encoder.activate(input)
            reconstructed = decoder.activate(bottleneck_output)
            for expected, output in zip(input, reconstructed):
                accuracy -= (expected - output) ** 2
        gen_behaviors[genome] = accuracy
        if accuracy > current_best_accuracy:
            current_best_accuracy = accuracy

        if len(all_behaviors) < k:
            # seeding behavior archive with initial behaviors
            all_behaviors.append([accuracy])

    archive = KDTree(np.array(all_behaviors))
    for genome, behavior in gen_behaviors.items():
        # get behavior distance and assign fitness
        distances, _ = archive.query([[behavior]])
        mean_distance = np.mean(distances)
        genome.fitness = mean_distance
        # archive if behavior distance exceeds threshold
        if mean_distance > archive_threshold:
            all_behaviors.append([behavior])

    # print the current best
    print(f'Current best error: {current_best_accuracy}')
    print(f'Current archive size: {len(all_behaviors)}')

config = neat.Config(
    neat.AutoencoderGenome, neat.DefaultReproduction,
    neat.DefaultSpeciesSet, neat.DefaultStagnation,
    'test-autoencoder-novelty.cfg'
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