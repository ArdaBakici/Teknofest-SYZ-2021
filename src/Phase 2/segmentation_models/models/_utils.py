from keras_applications import get_submodules_from_kwargs
import numpy as np

def freeze_model(model, **kwargs):
    """Set all layers non trainable, excluding BatchNormalization layers"""
    _, layers, _, _ = get_submodules_from_kwargs(kwargs)
    for layer in model.layers:
        layer.trainable = False
    return


def get_layer_number(model, layer_name):
    """
    Help find layer in Keras model by name
    Args:
        model: Keras `Model`
        layer_name: str, name of layer

    Returns:
        index of layer

    Raises:
        ValueError: if model does not contains layer with such name
    """
    for i, l in enumerate(model.layers):
        if l.name == layer_name:
            return i
    raise ValueError('No layer with name {} in  model {}.'.format(layer_name, model.name))


def extract_outputs(model, layers, include_top=False):
    """
    Help extract intermediate layer outputs from model
    Args:
        model: Keras `Model`
        layer: list of integers/str, list of layers indexes or names to extract output
        include_top: bool, include final model layer output

    Returns:
        list of tensors (outputs)
    """
    layers_indexes = ([get_layer_number(model, l) if isinstance(l, str) else l
                      for l in layers])
    outputs = [model.layers[i].output for i in layers_indexes]

    if include_top:
        outputs.insert(0, model.output)

    return outputs


def reverse(l):
    """Reverse list"""
    return list(reversed(l))


def recompile(model):
    model.compile(model.optimizer, model.loss, model.metrics)    
    

def set_trainable(model):
    for layer in model.layers:
        layer.trainable = True
    recompile(model)


def to_tuple(x):
    if isinstance(x, tuple):
        if len(x) == 2:
            return x
    elif np.isscalar(x):
        return (x, x)

    raise ValueError('Value should be tuple of length 2 or int value, got "{}"'.format(x))
