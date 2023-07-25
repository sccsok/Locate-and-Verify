from torch import nn

from warnings import warn


def buildActivation(activation_name, **kwargs):
    if activation_name not in activation_switch:
        warn(f'[buildActivation] Unrecognized activation: {activation_name}. Return identity instead')
    return activation_switch.get(activation_name, _identity)(**kwargs)


def _tanh(**kwargs):
    return nn.Tanh()


def _sigmoid(**kwargs):
    return nn.Sigmoid()


def _relu(**kwargs):
    return nn.ReLU()


def _identity(**kwargs):
    return nn.Identity()


activation_switch = {
    'tanh': _tanh,
    'sigmoid': _sigmoid,
    'relu': _relu,
    None: _identity,
    'identity': _identity,
    'none': _identity,
}