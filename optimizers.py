from __future__ import absolute_import
# import six
# import copy
from six.moves import zip
from keras.optimizers import Optimizer

from keras import backend as K
# from .utils.generic_utils import serialize_keras_object
# from .utils.generic_utils import deserialize_keras_object

if K.backend() == 'tensorflow':
    import tensorflow as tf


class SGD_momentless(Optimizer):
    """Stochastic gradient descent optimizer.

    Includes support for momentum,
    learning rate decay, and Nesterov momentum.

    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter updates momentum.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, lr=0.01, decay=0., **kwargs):
        super(SGD_momentless, self).__init__(**kwargs)
        self.iterations = K.variable(0., name='iterations')
        self.lr = K.variable(lr, name='lr')
        # self.momentum = K.variable(momentum, name='momentum')
        self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        # self.nesterov = nesterov

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = []

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))
            self.updates .append(K.update_add(self.iterations, 1))

        # momentum
        # shapes = [K.get_variable_shape(p) for p in params]
        # moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations]  # + moments
        # for p, g, m in zip(params, grads, moments):
        for p, g in zip(params, grads):
            # v = self.momentum * m - lr * g  # velocity
            v = -lr * g  # velocity
            # self.updates.append(K.update(m, v))

            # if self.nesterov:
            #     new_p = p + self.momentum * v - lr * g
            # else:
            new_p = p + v

            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov}
        base_config = super(SGD_momentless, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

