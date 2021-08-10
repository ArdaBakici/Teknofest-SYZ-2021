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
sm.custom.Double_Unet(backbone_name="vgg19", encoder_weights=None, input_shape=(512,512,3))