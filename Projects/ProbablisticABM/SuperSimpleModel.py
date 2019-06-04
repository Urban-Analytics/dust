import os
import warnings
import time
import imageio
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import matplotlib.pyplot as plt
from matplotlib import colors

import scipy.stats as stats

# from sys import platform
# if platform == "linux" or platform == "linux2":
#     clear = 'clear'
# elif platform == "win32":
#     clear = 'cls'

warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

try:
    tf.enable_eager_execution()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
except ValueError:
    pass

# For updating belief.
truth_observation = tfd.Normal(loc=1, scale=1.)

SIZE = 1000
EXIT = 100.

# Declare plot outside of loop, no need to keep building.
fig, ax = plt.subplots()


def custom_stepper():
    truthy = True

    # Position transformable
    x = tfd.Normal(loc=1, scale=1.)

    # Speed
    v = tfd.Exponential(rate=1)
    v_sample = v.sample(SIZE)
    # Noise
    u = tfd.Normal(loc=0, scale=1)

    clear_output_folder()

    t = 1.0

    while True:
        shift = tf.math.multiply(v_sample, tf.Variable(t))
        shift = tf.math.add(shift, u.sample(SIZE))
        affine = tfp.bijectors.Affine(shift)

        x = tfd.TransformedDistribution(distribution=x,
                                        bijector=affine)

        plot_agent(x)

        if tf.cond(pred=tf.greater(tf.math.reduce_mean(x.sample(SIZE)), tf.constant(EXIT)),
                   true_fn=lambda: True,
                   false_fn=lambda: False):
            break

        print('{}/{}'.format(format(tf.math.reduce_mean(x.sample(SIZE)), '9.5'), EXIT))

    render_agent()


def plot_agent(x):
    sample = x.sample(SIZE)
    n, bins, patches = ax.hist(sample, bins=100, density=1)
    fracs = n / n.max()
    norm = colors.Normalize(fracs.min(), fracs.max())
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.inferno(norm(thisfrac))
        thispatch.set_facecolor(color)

    ax.set_ylim(0, .3)
    ax.set_xlim(EXIT * -.25, EXIT * 1.5)
    ax.set_xlabel('x')
    ax.set_ylabel('Density in Location')
    ax.set_title(r'Single Agent:  $\mu={}$, $\sigma={}$'.format(format(tf.math.reduce_mean(sample),
                                                                       '9.5'),
                                                                format(tf.math.reduce_std(sample),
                                                                       '9.5')))
    # Save the plot to disk.
    plt.savefig('output/{}.png'.format(time.time()))
    # Clear for the next plot.
    ax.clear()


def render_agent():
    files = sorted(os.listdir('output'))
    print('{} frames generated.'.format(len(files)))
    images = []
    for filename in files:
        images.append(imageio.imread('output/{}'.format(filename)))
    imageio.mimsave('outputGIFs/{}.mp4'.format(time.time()), images)


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
