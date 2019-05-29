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

SIZE = 1000
EXIT = 100.


def custom_stepper():
    # Position
    x = tfd.Normal(loc=1, scale=1.)
    x_sample = x.sample(SIZE)
    # Speed
    v = tfd.Exponential(rate=5)
    v_sample = v.sample(SIZE)
    # Noise
    u = tfd.Normal(loc=0, scale=1)

    clear_output_folder()

    t = [[2.0]]
    while True:
        x_sample = tf.math.add(x_sample, tf.math.multiply(v_sample, tf.Variable(t)))
        x_sample = tf.math.add(x_sample, u.sample(SIZE))

        ax = sb.distplot(x_sample)

        ax.set(xlim=(EXIT * -.5, EXIT * .5))
        # ax.set(ylim=(0, .25))
        plt.axvline(EXIT, 0, tf.reduce_max(x_sample))
        plt.axvline(tf.reduce_mean(x_sample)[0], 0, tf.reduce_max(x_sample))

        plt.savefig('output/{}.png'.format(time.time()))
        plt.clf()

        if tf.cond(pred=tf.greater(tf.reduce_mean(x_sample), tf.constant(EXIT)),
                   true_fn=lambda: True,
                   false_fn=lambda: False):
            break

    render_agent()


def render_agent():
    files = os.listdir('output')
    print('{} frames generated.'.format(len(files)))
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
