from tensorflow import keras
import numpy as np
#import matplotlib.pylot as plt
import tensorflow as tf
import segmentation_models as sm
sm.set_framework('tf.keras')
# segmentation_models could also use `tf.keras` if you do not have Keras installed
# or you could switch to other framework using `sm.set_framework('tf.keras')`
from segmentation_models.custom import doubleunet

#doubleunet.double_unet(encoder_weights=None)
#doubleunet.double_unet()
#sm.custom.Double_Unet(backbone_name="seresnext101", encoder_weights=None, input_shape=(1024,1024,3))
for i in sm.get_available_backbone_names():
    print(f"----------Trying backend {i}----------")
    model_double_unet = sm.custom.Double_Unet(backbone_name=i, encoder_weights=None, input_shape=(512,512,3))
    print(f"----------No error in backend {i}----------")
    del model_double_unet