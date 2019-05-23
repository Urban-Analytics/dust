import os, shutil
import time

import imageio

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import seaborn as sb
import matplotlib.pyplot as plt


try:
    tf.enable_eager_execution()
except ValueError:
    pass

truth_observation = tfd.Normal(loc=1, scale=1.)

EXIT = 20.


def custom_stepper():
    x = tfd.Normal(loc=1, scale=1.)
    states = []
    v = tfd.Exponential(rate=1)

    affine = tfp.bijectors.Affine(shift=v.sample(1))

    clear_output_folder()

    while True:
        x = tfd.TransformedDistribution(
            distribution=x,
            bijector=affine,
            name="Shift")

        states.append(x)

        ax = sb.distplot(x.sample(1000, name='x'))
        ax.set(xlim=(-1, EXIT + 1))
        ax.set(ylim=(0, 1.5))
        plt.savefig('output/{}.png'.format(time.time()))
        plt.clf()

        print(x.sample(100))

        if tf.cond(pred=tf.greater(x.sample(1), tf.constant(EXIT)),
                   true_fn=lambda: True,
                   false_fn=lambda: False):
            break

    render_agent()


def render_agent():
    files = os.listdir('output')
    print(files)
    images = []
    for filename in files:
        images.append(imageio.imread('output/{}'.format(filename)))
    imageio.mimsave('outputGIFs/{}.gif'.format(time.time()), images)


def clear_output_folder():
    folder = 'output'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    custom_stepper()
