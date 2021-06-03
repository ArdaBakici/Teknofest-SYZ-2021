import os
os.environ['KERAS_BACKEND'] = "plaidml.keras.backend"
from keras.layers import BatchNormalization, LeakyReLU, SpatialDropout2D, Dropout
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Concatenate
from keras.models import Model
from keras.optimizers import Adam, SGD
from .custom_layers import Softmax4D
from .custom_metrics import multiclass_balanced_cross_entropy, multiclass_dice_coef_metric


class Models:
    def __init__(self, model_train, model_save):
        self.model_train = model_train
        self.model_main = model_save
        self.multigpu_train = False

def create_unet_model(N_classes, input_shape=(None, None, 1), dropout_rate=0.24, learning_rate=1e-5):
    """
    Implementation of Unet mode for multiclass semantic segmentation
    :param N_classes: Number of classes of segmentation map
    :param input_shape: input image shape
    :param dropout_rate: dropout  rate
    :return: a tuple of two models, first element is model to train and second is model to save
    """

    # make sure the sizes are divisible by 16
    if(input_shape[0] is not None):  assert 16 * (input_shape[0] // 16) == input_shape[0], 'invalid dimension 0'
    if( input_shape[1]  is not None): assert 16 * (input_shape[1] // 16) == input_shape[1], 'invalid dimension 1'

    in_image = Input(shape=input_shape)

    conv0 = Conv2D(32, (3, 3), activation='relu', name='conv1_0', padding='same')(in_image)

    conv1, x = conv_block_down(32, dropout_rate=dropout_rate ) (conv0)
    conv2, x  = conv_block_down(64, dropout_rate=dropout_rate ) (x)
    conv3, x  = conv_block_down(128, dropout_rate=dropout_rate )(x)
    conv4, x  = conv_block_down(256, dropout_rate=dropout_rate )(x)

    x = conv_block(512, dropout_rate=dropout_rate ) (x)

    x = deconv_block(512, skip_layer=conv4, dropout_rate=dropout_rate ) (x)
    x = deconv_block(256, skip_layer=conv3, dropout_rate=dropout_rate ) (x)
    x = deconv_block(128, skip_layer=conv2, dropout_rate=dropout_rate ) (x)
    x = deconv_block(64,  skip_layer=conv1, dropout_rate=dropout_rate ) (x)


    outp_logit = Conv2D(N_classes, (1, 1), activation='linear', padding='same', name='logit')(x)
    outp_softmax = Softmax4D(axis=3, name='segmap')(outp_logit)


    model_train = Model(inputs=in_image, outputs=[outp_logit,outp_softmax])
    model_save = Model(inputs=in_image, outputs=[outp_softmax])

    #if last channel is background
    if(N_classes <=5):
        class_indices = list(range(N_classes))[:-1] #except last one which is background
        metrics_classwise=[]
        for c in class_indices:
            fc = multiclass_dice_coef_metric(from_logits=True, class_index=c)
            fc.__name__='dmc'+str(c)
            metrics_classwise.append(fc)

        metrics = {'logit': metrics_classwise}
    else:
        metrics = {'logit': [multiclass_dice_coef_metric(from_logits=True)]} #all classes

    model_train.compile(optimizer=SGD(lr=learning_rate),
                        loss={'logit': multiclass_balanced_cross_entropy(from_logits=True, P=5)},
                        metrics=metrics)

    return Models(model_train, model_save)



def conv_block(num_filters, filter_size=(3,3), name=None, padding='same',dropout_rate=None, training=True):
    def f(x):

        x = Conv2D(num_filters, filter_size, name=name, padding=padding, kernel_initializer = 'he_normal')(x)
        x = BatchNormalization()(x, training=training)
        x = LeakyReLU(alpha=0.3)(x)
        if (dropout_rate is not None): x = SpatialDropout2D(dropout_rate)(x)
        return x

    return f


def conv_block_down(num_filters, filter_size=(3,3), name=None, padding='same', dropout_rate=None, training=True):
    def f(x):
        x = conv_block(num_filters, filter_size=filter_size, name=name, padding=padding,
                       dropout_rate=dropout_rate, training=training)(x)

        tran_down = MaxPooling2D(pool_size=(2, 2))(x)
        return x, tran_down

    return f

def deconv_block(num_filters, skip_layer, filter_size=(3,3), strides=(2, 2),name=None, padding='same', dropout_rate=None, training=True):
    def f(x):
        x = Conv2DTranspose(num_filters, filter_size, strides=strides, padding=padding, kernel_initializer ='he_normal')(x)
        x = Concatenate(axis=3)([x, skip_layer])
        x = Conv2D(int(num_filters), filter_size, name=name, padding=padding)(x)
        x = BatchNormalization()(x, training=training)
        x = LeakyReLU(alpha=0.3)(x)
        if (dropout_rate is not None): x = SpatialDropout2D(dropout_rate)(x)
        return x

    return f


if (__name__ == '__main__'):
    IMG_W, IMG_H = 256, 320

    input_shape = (IMG_H, IMG_W, 1)
    m1, m2 = create_unet_model(N_classes=5, input_shape=input_shape)
    # model.compile(optimizer=Adam(lr=1e-5),
    #              loss={'logit': multiclass_balanced_cross_entropy(from_logits=True)},
    #              metrics={'segmap': dice_coef})


    print(m1.summary())

