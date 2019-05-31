import os
import warnings
import time
import imageio
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

try:
    tf.enable_eager_execution()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
except ValueError:
    pass

truth_observation = tfd.Normal(loc=1, scale=1.)
print(tf.test.is_gpu_available())


SIZE = 1000
EXIT = 100.

fig, ax = plt.subplots()


def custom_stepper():
    tog = True

    # Position
    x = tfd.Normal(loc=1, scale=1.)
    x_sample = x.sample(SIZE)
    # Speed
    v = tfd.Exponential(rate=1)
    v_sample = v.sample(SIZE)
    # Noise
    u = tfd.Normal(loc=0, scale=0.5)

    clear_output_folder()

    t = [[1.0]]
    while True:
        x_sample = tf.math.add(x_sample, tf.math.multiply(v_sample, tf.Variable(t)))
        x_sample = tf.math.add(x_sample, u.sample(SIZE))

        os.system('clear')
        if tog:
            print('\\Working/')
            tog = False
        else:
            print('/Working\\')
            tog = True

        print(tf.math.reduce_mean(x_sample).numpy())
        ax.hist(x_sample, range=[0, EXIT * 1.5], bins=100, label=str(tf.math.reduce_mean(x_sample)))
        ylim = SIZE * .5
        ax.set_ylim(0, ylim)
        ax.set_xlabel('x')
        ax.set_ylabel('Probability of Location')

        plt.savefig('output/{}.png'.format(time.time()))
        ax.clear()

        if tf.cond(pred=tf.greater(tf.math.reduce_mean(x_sample), tf.constant(EXIT)),
                   true_fn=lambda: True,
                   false_fn=lambda: False):
            break

    render_agent()


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
