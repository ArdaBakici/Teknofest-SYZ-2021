# This is the python file to register your custom backbone/model to segmentation models framework.
# If you don't register your model here segmentation models framework won't be able to find your model
import functools
from custom_model_template import model as modelFunc
from ..__init__ import inject_global_submodules
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
    # Put here functions to your models
    modelFunc = inject_global_submodules(modelFunc) # put your model function into inject_global_submodels
