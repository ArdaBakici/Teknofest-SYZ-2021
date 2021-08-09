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
    def wrapper(*args, **kwargs):
        kwargs['backend'] = _KERAS_BACKEND
        kwargs['layers'] = _KERAS_LAYERS
        kwargs['models'] = _KERAS_MODELS
        kwargs['utils'] = _KERAS_UTILS
        return func(*args, **kwargs)

    return wrapper


# Add your custom backbones to here
custom_backbone_list = {
    # backbone name : [backbone function, backbone preprocessing] example:
    # 'resnet101' : [resnet.resnet101, resnet.preprocessinput],
}

custom_default_feature_layers = {
    # This file must be filled for each backbone to be used in a model.
    # This dictionary indicates to model for taking features from which channels
    # Syntax should be in the following order
    # List of layers to take features from backbone in the following order:
    # (x16, x8, x4, x2, x1) - `x4` mean that features has 4 times less spatial
    # resolution (Height x Width) than input image.
    # example:
    # 'vgg16': ('block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2'),
    # 'resnext50': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    # 'senet154': (6884, 1625, 454, 12),
}

class Custom_Registry:
    def __init__(self, backend, layers, models, utils):
        _KERAS_BACKEND = backend
        _KERAS_LAYERS = layers
        _KERAS_MODELS = models
        _KERAS_UTILS = utils
        
    # Put here functions to your models
    Double_Unet = inject_global_submodules(double_unet) # put your model function into inject_global_submodels
