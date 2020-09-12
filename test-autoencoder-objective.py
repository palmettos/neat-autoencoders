import neat
import numpy as np
from sklearn.neighbors import KDTree
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider


dimensions = 6

inputs = [
    [1., 0., 1., 0., 1., 0.],
    [0., 1., 0., 1., 0., 1.],
    [1., 1., 0., 0., 1., 0.],
    [0., 1., 1., 0., 1., 0.],
    [0., 1., 0., 1., 0., 1.],
    [1., 1., 1., 0., 0., 0.]
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

input = inputs[0]
bottleneck_output = encoder.activate(input)
reconstructed = decoder.activate(bottleneck_output)


fig, ax = plt.subplots()
im = ax.imshow(np.array([reconstructed]))

axcolor = 'lightgoldenrodyellow'
decoder_input_axes = []
decoder_input_sliders = []
for i, out in enumerate(bottleneck_output):
    y_pos = (i + 1) * 0.1
    decoder_input_axes.append(plt.axes([0.25, y_pos, 0.65, 0.03], facecolor=axcolor))
    decoder_input_sliders.append(Slider(decoder_input_axes[i], f'bottleneck_input{str(i)}', 0.0, 1.0, valinit=out))

def create_update_func(i):
    def update(val):
        bottleneck_output[i] = val
        print(bottleneck_output)
        im.set_data(np.array([decoder.activate(bottleneck_output)]))
    return update

for i, slider in enumerate(decoder_input_sliders):
    slider.on_changed(create_update_func(i))

plt.show()