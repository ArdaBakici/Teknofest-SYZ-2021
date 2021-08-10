# This is the python file to register your custom backbone/model to segmentation models framework.
# If you don't register your model here segmentation models framework won't be able to find your model
import functools
from .doubleunet import double_unet

_KERAS_BACKEND = None
_KERAS_LAYERS = None
_KERAS_MODELS = None
_KERAS_UTILS = None

def inject_global_submodules(func):
    @functools.wraps(func)
    def wrapper(self:"Custom_Registry", *args, **kwargs):
        kwargs['backend'] = _KERAS_BACKEND
        kwargs['layers'] = _KERAS_LAYERS
        kwargs['models'] = _KERAS_MODELS
        kwargs['utils'] = _KERAS_UTILS
        return func(*args, **kwargs)

    return wrapper

class Custom_Registry:
    def __init__(self, backend, layers, models, utils):
        global _KERAS_BACKEND, _KERAS_LAYERS, _KERAS_MODELS, _KERAS_UTILS
        _KERAS_BACKEND = backend
        _KERAS_LAYERS = layers
        _KERAS_MODELS = models
        _KERAS_UTILS = utils
        
    # Put here functions to your models
    Double_Unet = inject_global_submodules(double_unet) # put your model function into inject_global_submodels
