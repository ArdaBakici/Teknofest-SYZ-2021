from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications import *
from ..models._common_blocks import Conv2dBn
from ..models._utils import freeze_model
from ..backbones.backbones_factory import Backbones
from keras_applications import get_submodules_from_kwargs

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
#  Building Blocks
# ---------------------------------------------------------------------

def Conv3x3BnReLU(filters, use_batchnorm, name=None):
    kwargs = get_submodules()

    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            use_batchnorm=use_batchnorm,
            name=name,
            **kwargs
        )(input_tensor)

    return wrapper

def output_block(class_num, activation):
    def wrapper(input_tensor):
        x = Conv2D(class_num, (1, 1), padding="same")(input_tensor)
        x = Activation(activation, dtype='float32')(x)
        return x
    return wrapper

# ---------------------------------------------------------------------
# Advanced Blocks
# ---------------------------------------------------------------------

def SqueezeAndExciteBlock(name, stage, ratio=8):
    kwargs = get_submodules()
    glob_av_name = f"{name}se_{stage}_global_average"
    reshape_name= f"{name}se_{stage}_reshaper"
    dense_1_name= f"{name}se_{stage}_dense_a"
    dense_2_name= f"{name}se_{stage}_dense_b"

    def wrapper(input_tensor):
        channel_axis = -1 if backend.image_data_format() == 'channels_last' else 1
        filters = input_tensor.shape[channel_axis]
        se_shape = (1, 1, filters)
        se = GlobalAveragePooling2D(name=glob_av_name)(input_tensor)
        se = Reshape(se_shape, name=reshape_name)(se)
        se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False, name=dense_1_name)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False, name=dense_2_name)(se)

        x = Multiply()([input_tensor, se])
        return x

    return wrapper

# ---------------------------------------------------------------------
# Decoder Blocks
# ---------------------------------------------------------------------

def DecoderUpsamplingX2Block(filters, order, stage, use_batchnorm=False):
    # 2 stage decoders
    # TODO check all names since there is 2 decoders
    up_name = f'decoder_upsampling_{order}_stage{stage}_upsampling'
    conv1_name = f'decoder_upsampling_{order}_stage{stage}a'
    conv2_name = f'decoder_upsampling_{order}_stage{stage}b'
    concat_name = f'decoder_upsampling_{order}_stage{stage}_concat'

    concat_axis = -1 if backend.image_data_format() == 'channels_last' else 1

    if order == 1:
        def wrapper(input_tensor, skip=None):
            x = layers.UpSampling2D(size=2, name=up_name)(input_tensor)

            if skip is not None:
                x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])

            x = Conv3x3BnReLU(filters, use_batchnorm, name=conv1_name)(x)
            x = Conv3x3BnReLU(filters, use_batchnorm, name=conv2_name)(x)
            x = SqueezeAndExciteBlock("UpsamplingDecoder1", stage)(x)

            return x
    elif order == 2:
        def wrapper(input_tensor, skip1=None, skip2=None):
            x = layers.UpSampling2D(size=2, name=up_name)(input_tensor)

            if skip1 is not None:
                if skip2 is not None:
                    x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip1, skip2])
                else:
                    x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip1])
            elif skip2 is not None:
                x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip2])

            x = Conv3x3BnReLU(filters, use_batchnorm, name=conv1_name)(x)
            x = Conv3x3BnReLU(filters, use_batchnorm, name=conv2_name)(x)
            x = SqueezeAndExciteBlock("UpsamplingDecoder2", stage)(x)

            return x
    else:
        ValueError("Encoder order can either be 1 or 2")
    return wrapper


def DecoderTransposeX2Block(filters, order, stage, use_batchnorm=False):
    transp_name = f'decoder_transpose_{order}_stage{stage}a_transpose'
    bn_name = f'decoder_transpose_{order}_stage{stage}a_bn'
    relu_name = f'decoder_transpose_{order}_stage{stage}a_relu'
    conv_block_name = f'decoder_transpose_{order}_stage{stage}b'
    concat_name = f'decoder_transpose_{order}_stage{stage}_concat'
    concat_axis = bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if order == 1:
        def layer(input_tensor, skip=None):

            x = layers.Conv2DTranspose(
                filters,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding='same',
                name=transp_name,
                use_bias=not use_batchnorm,
            )(input_tensor)

            if use_batchnorm:
                x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

            x = layers.Activation('relu', name=relu_name)(x)

            if skip is not None:
                x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])

            x = Conv3x3BnReLU(filters, use_batchnorm, name=conv_block_name)(x)
            x = SqueezeAndExciteBlock("TransposeDecoder1", stage)(x)

            return x
    elif order == 2:
        def layer(input_tensor, skip1=None, skip2=None):

            x = layers.Conv2DTranspose(
                filters,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding='same',
                name=transp_name,
                use_bias=not use_batchnorm,
            )(input_tensor)

            if use_batchnorm:
                x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

            x = layers.Activation('relu', name=relu_name)(x)

            if skip1 is not None:
                if skip2 is not None:
                    x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip1, skip2])
                else:
                    x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip1])
            elif skip2 is not None:
                x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip2])

            x = Conv3x3BnReLU(filters, use_batchnorm, name=conv_block_name)(x)
            x = SqueezeAndExciteBlock("TransposeDecoder2", stage)(x)

            return x
    else:
        ValueError("Encoder order can either be 1 or 2")

    return layer

# ---------------------------------------------------------------------
# Encoder Blocks
# ---------------------------------------------------------------------

def ConvBlock_SE(filters, use_batchnorm, name, stage):
    kwarg = get_submodules()
    conv_block1_name = f'{name}_convblock_se_convbnrelu_stage{stage}a'
    conv_block2_name = f'{name}_convblock_se_convbnrelu_stage{stage}b'
    squeeze_name = f'{name}_convblock_se_seblock_stage{stage}'
    def wrapper(input_tensor):
        x = Conv3x3BnReLU(filters, use_batchnorm, conv_block1_name)(input_tensor)
        x = Conv3x3BnReLU(filters, use_batchnorm, conv_block2_name)(x)
        x = SqueezeAndExciteBlock(squeeze_name, stage)(x)
        return x

    return wrapper
# ---------------------------------------------------------------------
#  Unet ASPP
# ---------------------------------------------------------------------
def ASPP(filters, name):
    kwargs = get_submodules()
    average_pooling_name = f"ASPP_{name}_average_pool"
    conv_1_name = f"ASPP_{name}_conv_a"
    bn_1_name = f"ASPP_{name}_bn_a"
    activation_1_name = f"ASPP_{name}_activation_a"
    upsample_name = f"ASPP_{name}_upsampling"
    conv_2_name = f"ASPP_{name}_conv_b"
    bn_2_name = f"ASPP_{name}_bn_b"
    activation_2_name = f"ASPP_{name}_activation_b"
    conv_3_name = f"ASPP_{name}_conv_c"
    bn_3_name = f"ASPP_{name}_bn_c"
    activation_3_name = f"ASPP_{name}_activation_c"
    conv_4_name = f"ASPP_{name}_conv_d"
    bn_4_name = f"ASPP_{name}_bn_d"
    activation_4_name = f"ASPP_{name}_activation_d"
    conv_5_name = f"ASPP_{name}_conv_e"
    bn_5_name = f"ASPP_{name}_bn_e"
    activation_5_name = f"ASPP_{name}_activation_e"
    concat_name = f"ASPP_{name}_concat"
    conv_last_name = f"ASPP_{name}_conv_last"
    bn_last_name = f"ASPP_{name}_bn_last"
    activation_last_name = f"ASPP_{name}_activation_last"
    

    def wrapper(x):
        shape = x.shape

        y1 = AveragePooling2D(pool_size=(shape[1], shape[2]), name=average_pooling_name)(x)
        y1 = Conv2D(filters, 1, padding="same", name=conv_1_name)(y1)
        y1 = BatchNormalization(name=bn_1_name)(y1)
        y1 = Activation("relu", name=activation_1_name)(y1)
        y1 = UpSampling2D((shape[1], shape[2]), interpolation='bilinear', name=upsample_name)(y1)

        y2 = Conv2D(filters, 1, dilation_rate=1, padding="same", use_bias=False, name=conv_2_name)(x)
        y2 = BatchNormalization(name=bn_2_name)(y2)
        y2 = Activation("relu", name=activation_2_name)(y2)

        y3 = Conv2D(filters, 3, dilation_rate=6, padding="same", use_bias=False, name=conv_3_name)(x)
        y3 = BatchNormalization(name=bn_3_name)(y3)
        y3 = Activation("relu", name=activation_3_name)(y3)

        y4 = Conv2D(filters, 3, dilation_rate=12, padding="same", use_bias=False, name=conv_4_name)(x)
        y4 = BatchNormalization(name=bn_4_name)(y4)
        y4 = Activation("relu", name=activation_4_name)(y4)

        y5 = Conv2D(filters, 3, dilation_rate=18, padding="same", use_bias=False, name=conv_5_name)(x)
        y5 = BatchNormalization(name=bn_5_name)(y5)
        y5 = Activation("relu", name=activation_5_name)(y5)

        y = Concatenate(name=concat_name)([y1, y2, y3, y4, y5])

        y = Conv2D(filters, 1, dilation_rate=1, padding="same", use_bias=False, name=conv_last_name)(y)
        y = BatchNormalization(name=bn_last_name)(y)
        y = Activation("relu", name=activation_last_name)(y)

        return y

    return wrapper

# ---------------------------------------------------------------------
#  Unet Encoder
# ---------------------------------------------------------------------

def encoder2(encoder_block, encoder_filters, use_batchnorm=True):
    def wrapper(x):
        skip_connections = []
        for i in range(len(encoder_filters)):
            x = encoder_block(encoder_filters[i], use_batchnorm, "encoder2", i)(x)
            skip_connections.append(x)
            x = MaxPool2D((2, 2))(x)
        return x, skip_connections
    return wrapper

# ----------------------------------------------------------------------
#  Unet Decoder
# ----------------------------------------------------------------------

def decoder1(decoder_block_1, decoder_filters, skip_connections, use_batchnorm=True):
    kwargs = get_submodules()

    def wrapper(x):
        for i in range(len(decoder_filters)):
            if i < len(skip_connections):
                skip = skip_connections[i]
            else:
                skip = None

            x = decoder_block_1(decoder_filters[i], order=1, stage=i, use_batchnorm=use_batchnorm)(x, skip)
        return x

    return wrapper

def decoder2(decoder_block_2, decoder_filters, skips_1, skips_2, use_batchnorm=True):
    kwargs = get_submodules()

    def wrapper(x):
        for i in range(len(decoder_filters)):
            if i < len(skips_1):
                skip_1 = skips_1[i]
            else:
                skip_1 = None

            if i < len(skips_1):
                skip_2 = skips_2[i]
            else:
                skip_2 = None
            x = decoder_block_2(decoder_filters[i], order=2, stage=i, use_batchnorm=use_batchnorm)(x, skip_1, skip_2)
        return x
    return wrapper

# --------------------------------------o--------------------------------------

def build_double_unet(
            backbone,
            decoder_block_1,
            decoder_block_2,
            skip_connection_layers,
            decoder_filters,
            encoder_filters,
            encoder_block,
            aspp_filter_size,
            classes,
            activation,
            use_center_block,
            use_batchnorm):
    print("Settings initialized. Starting to build model.")
    input = backbone.input
    x = backbone.output # size = input_shape / 2^5
    # Artitechture of double unet
    # backbone -> ASPP -> Decoder with skips from backbone -> Output Block -> Multiply with backbone -> Encoder 2 -> ASPP -> Decoder 2 with skips from both encoder 1 and 2-> Output Block -> Concatenate
    # extract skip connections for encoder 1

    # from last to first
    skips_enc_1 = ([backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in skip_connection_layers])

    # for vgg models ending with maxpooling
    # we can either put a center block or do not get the maxpooling block
    # if we do not get the maxpooling block we do not need to skip the last layer since its output is direcly used
    if isinstance(backbone.layers[-1], layers.MaxPooling2D):
        if (use_center_block):
            x = Conv3x3BnReLU(512, use_batchnorm, name='center_block1')(x)
            x = Conv3x3BnReLU(512, use_batchnorm, name='center_block2')(x)
        else:
            x = backbone.layers[-2].output
            # for vgg models also skip last layer if not don't skip it
            del skips_enc_1[0]
    x = ASPP(aspp_filter_size, name="1")(x) # no size change
    print("ASPP-1 initialized...")
    x = decoder1(decoder_block_1, decoder_filters, skips_enc_1, use_batchnorm)(x) # size = x * 2^block_amount
    print("Decoder-1 initialized...")
    output1 = output_block(classes, activation)(x) # no size change
    print("Output-1 initialized...")
    x = input * output1 # no size change
    print(f"output is {output1}")
    print(f"x is {x}")
    x, skips_enc_2 = encoder2(encoder_block, encoder_filters, use_batchnorm)(x) # size = x / 2^block_amount
    skips_enc_2.reverse()
    print("Encoder-2 initialized...")
    x = ASPP(aspp_filter_size, name="2")(x) # no size change
    print("ASPP-2 initialized...")
    x = decoder2(decoder_block_2, decoder_filters, skips_enc_1, skips_enc_2, use_batchnorm)(x) # size = x * 2^block_amount
    print("Decoder-2 initialized...")
    output2 = output_block(classes, activation)(x) # no size change
    print("Output-2 initialized...")
    output = Concatenate()([output1, output2]) # no size change but channel change
    model = Model(input, output)
    print("Model initialization complete.")
    return model

def double_unet(backbone_name='vgg19',
                input_shape=(None, None, 3),
                classes=3,
                activation='softmax',
                weights=None,
                encoder_weights='imagenet',
                encoder_freeze=False,
                encoder_features='default',
                use_center_block=True,
                aspp_filter_size=64,
                encoder_2_block_type='ConvBlock_SE',
                decoder_1_block_type='upsampling',
                decoder_2_block_type='upsampling',
                encoder_filters=(16, 32, 64, 128, 256),
                decoder_filters=(256, 128, 64, 32, 16),
                use_batchnorm=True,
                **kwargs):

    """ Double Unet is a fully convolution neural network for semantic segmantation.

    Args:
        backbone_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        input_shape: shape of input data/image ``(H, W, C)``, in general
            case you do not need to set ``H`` and ``W`` shapes, just pass ``(None, None, C)`` to make your model be
            able to process images af any size, but ``H`` and ``W`` of input images should be divisible by factor ``32``.
        classes: a number of classes for output (output shape - ``(h, w, classes)``).
        activation: name of one of ``keras.activations`` for last model layer
            (e.g. ``sigmoid``, ``softmax``, ``linear``).
        weights: optional, path to model weights.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        encoder_freeze: if ``True`` set all layers of encoder (backbone model) as non-trainable.
        encoder_features: a list of layer numbers or names starting from top of the model.
            Each of these layers will be concatenated with corresponding decoder block. If ``default`` is used
            layer names are taken from ``DEFAULT_SKIP_CONNECTIONS``.
        use_center_block: Use a center block for models ending with maxpooling. If it is set to false last maxpooling layers won't be included. (this includes vgg models)
        aspp_filter_size: Filter size for ASPP part of the model.
        encoder_2_block_type: Encoder block to be used. Currently only option is ConvBlock_SE.
        decoder_1_block_type: Type of the first decoder. One of blocks with following layers structure:

            - `upsampling`:  ``UpSampling2D`` -> ``Conv2D`` -> ``Conv2D``
            - `transpose`:   ``Transpose2D`` -> ``Conv2D``

        decoder_2_block_type: Type of the second decoder.
        encoder_filters: list of numbers of ``Conv2D`` layer filters in encoder blocks
        decoder_filters: list of numbers of ``Conv2D`` layer filters in decoder blocks
        use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used.

    Returns:
        ``keras.models.Model``: **Double Unet**
    """
    print("Starting up Double Unet initialization...\nProcessing settings...")
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if decoder_1_block_type == 'upsampling':
        decoder_1_block = DecoderUpsamplingX2Block
    elif decoder_1_block_type == 'transpose':
        decoder_1_block = DecoderTransposeX2Block
    else:
        raise ValueError('Decoder 1 block type should be in ("upsampling", "transpose"). '
                         'Got: {}'.format(decoder_1_block_type))
                        
    if decoder_2_block_type == 'upsampling':
        decoder_2_block = DecoderUpsamplingX2Block
    elif decoder_2_block_type == 'transpose':
        decoder_2_block = DecoderTransposeX2Block
    else:
        raise ValueError('Decoder 2 block type should be in ("upsampling", "transpose"). '
                         'Got: {}'.format(decoder_2_block_type))
    backbone = Backbones.get_backbone(
        backbone_name,
        input_shape=input_shape,
        weights=encoder_weights,
        include_top=False,
        **kwargs,
    )

    if encoder_features == 'default':
        if backbone_name == 'vgg16' or backbone_name == 'vgg19':
            encoder_features = Backbones.get_feature_layers(backbone_name, n=5)
        else:
            encoder_features = Backbones.get_feature_layers(backbone_name, n=4) # Not enough for vgg

    if encoder_2_block_type == 'ConvBlock_SE':
        encoder_2_block = ConvBlock_SE
    else:
        raise ValueError('Encoder 2 block type should be in ("ConvBlock_SE").'
                         f'Got: {encoder_2_block_type}')

    model = build_double_unet(
                backbone=backbone,
                decoder_block_1=decoder_1_block,
                decoder_block_2=decoder_2_block,
                skip_connection_layers=encoder_features,
                decoder_filters=decoder_filters,
                encoder_filters=encoder_filters,
                encoder_block=encoder_2_block,
                aspp_filter_size=aspp_filter_size,
                classes=classes,
                activation=activation,
                use_center_block=use_center_block,
                use_batchnorm=use_batchnorm)

    # lock encoder weights for fine-tuning
    if encoder_freeze:
        freeze_model(backbone, **kwargs)

    # loading model weights
    if weights is not None:
        model.load_weights(weights)

    return model