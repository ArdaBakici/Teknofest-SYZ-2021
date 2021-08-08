# This file provides a template for creating custom models for segmantation models framework. This
# file shouldn't be overridden instead it should be copied and used

from tensorflow import keras

backend = None
layers = None
models = None
keras_utils = None


# ---------------------------------------------------------------------
#  Utility functions
# ---------------------------------------------------------------------

def get_submodules():
    return {
        'backend': backend,
        'models': models,
        'layers': layers,
        'utils': keras_utils,
    }

    
# ---------------------------------------------------------------------
#  Blocks
# ---------------------------------------------------------------------



def build_model() -> keras.models.Model:
    """ This function is used for building your model. Function must return a keras.Model type. Input of the model
        must be the input of the backbone and all the function layers must be put on top of the backbone output """
    pass

def model() -> keras.models.Model:
    """ This function is the main called function and using the helper functions defined above you must create
        a keras model and return it """
    pass