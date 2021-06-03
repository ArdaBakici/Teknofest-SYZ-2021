import os
os.environ['KERAS_BACKEND'] = "plaidml.keras.backend"
from keras.models import model_from_json
import os
from  .custom_layers import Softmax4D


def load_tfkeras_model( model_directory, file_name_prefix, model=None, custom_objects={}):
    """
    Loads  model saved as hd5 file
    :param model_directory:
    :param file_name_prefix:
    :param model:
    :param custom_objects:
    :return:
    """
    if("Softmax4D" not in custom_objects):
        custom_objects["Softmax4D"] = Softmax4D


    json = os.path.join(os.getcwd(), model_directory, file_name_prefix + '.json')
    with open(json) as j:
        json_string = j.read()

    if (model is None):
        amodel = model_from_json(json_string, custom_objects=custom_objects)
    else:
        amodel = model

    amodel.load_weights(os.path.join(model_directory, file_name_prefix + '.hd5'))
    return amodel
