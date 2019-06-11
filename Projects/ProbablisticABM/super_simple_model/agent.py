import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import os

try:
    tf.enable_eager_execution()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
except ValueError:
    pass


class Agent:

    def __init__(self, x, y, **kwargs):
        self.noise = kwargs['noise'] if kwargs['noise'] else None
        self.sample_size = kwargs['sample_size'] if kwargs['sample_size'] else 1
        self.x = x
        self.y = y
        self.t = 1.0
        self.xy = tfd.Normal(loc=[self.x, self.y], scale=[.1, .1])
        self.v = tfd.Normal(loc=[1., 0.], scale=[1., 1.])
        self.u = tfd.Normal(loc=[0, 0], scale=self.noise if self.noise else 0)
        self.v_sample = self.v.sample(self.sample_size)

    def step(self):
            step = tf.math.multiply(self.v_sample, tf.Variable(self.t))
            if self.noise:
                step = tf.math.add(step, self.u.sample(self.sample_size))
            affine = tfp.bijectors.Affine(step)
            self.xy = tfd.TransformedDistribution(distribution=self.xy,
                                                  bijector=affine)

    def get_sample_position(self):
        xy_sample = self.xy.sample(self.sample_size)
        return xy_sample[..., 0], xy_sample[..., 1]
