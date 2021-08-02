# This file provides a template for creating custom backbone for segmantation models framework. This
# file shouldn't be overridden instead it should be copied and used

import os
import collections

from tensorflow import keras

# This is the main function to generate the backbone. This function will be used in the different confugiration generating functions
# Must return keras.Model
def backbone(model_params,
            input_tensor=None,
            input_shape=None,
            include_top=True,
            classes=1000,
            weights='imagenet',
            **kwargs) -> keras.models.Model:
    pass

# Below functions are different types of configurations for your backbone
# they can be renamed and should call your main backbone creation function and they should return keras.Model

def model21(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True, **kwargs) -> keras.models.Model:
    pass

def model51(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True, **kwargs) -> keras.models.Model:
    pass

def model101(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True, **kwargs) -> keras.models.Model:
    pass


# This function will be called upon preprocessing step. Put here your preprocessing steps
def preprocess_input(x, **kwargs):
    pass