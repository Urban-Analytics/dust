import os
import warnings
import time
import imageio
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import matplotlib.pyplot as plt
import numpy as np

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
NOISE = 10
ENVIRONMENT = (1000, 1000)

# Declare plot outside of loop, no need to keep building.
plt.figure(figsize=(8, 8))


def custom_stepper():
    clear_output_folder()

    xy = tfd.Normal(loc=[1., ENVIRONMENT[1]*.5],
                    scale=[.1, .1])
    v = tfd.Normal(loc=[2., 0.],
                   scale=[1., 1.])
    u = tfd.Normal(loc=[0, 0], scale=NOISE)

    t = 1.0

    v_sample = v.sample(SIZE)

    i = 0
    while True:
        xy_sample = xy.sample(SIZE)
        x = xy_sample[..., 0]
        y = xy_sample[..., 1]

        plot_agent(x, y)

        plt.savefig('output/{}.png'.format(time.time()))
        plt.clf()

        step = tf.math.multiply(v_sample, tf.Variable(t))
        step = tf.math.add(step, u.sample(SIZE))
        affine = tfp.bijectors.Affine(step)
        xy = tfd.TransformedDistribution(distribution=xy,
                                         bijector=affine)



        if i > 200:
            break
        i = i + 1

        # if tf.cond(pred=tf.greater(tf.math.reduce_mean(x.sample(SIZE)), tf.constant(EXIT)),
        #            true_fn=lambda: True,
        #            false_fn=lambda: False):
        #     break

        # print('{}/{}'.format(format(tf.math.reduce_mean(x.sample(SIZE)), '9.5'), EXIT))

    render_agent()


def plot_agent(x, y):
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)

    ax_scatter.scatter(x, y, s=1, label='Potential Locations')
    ax_scatter.scatter(np.median(x), np.median(y), s=10, c='red', marker='x', label='Highest Probable Location')

    ax_scatter.set_xlim((ENVIRONMENT[0] * -.2, ENVIRONMENT[0]))
    ax_scatter.set_ylim((0, ENVIRONMENT[1]))
    ax_scatter.legend(loc='upper right')
    ax_scatter.grid(True)

    n, bins, patches = ax_histx.hist(x, bins=100, density=1)
    build_heatmap(n, patches)
    n, bins, patches = ax_histy.hist(y, bins=100, orientation='horizontal', density=1)
    build_heatmap(n, patches)

    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())


def build_heatmap(n, patches):
    fracs = n / n.max()
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.inferno(np.linalg.norm(thisfrac))
        thispatch.set_facecolor(color)


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
