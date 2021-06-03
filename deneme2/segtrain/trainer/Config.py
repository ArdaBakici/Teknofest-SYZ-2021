import os
os.environ['KERAS_BACKEND'] = "plaidml.keras.backend"


class Config:
    """Base Configuration with default values.

    """
    # Give the configuration a recognizable name
    # This name will be used in logging the training and saving the model
    # Useful when different experiments with different settings
    NAME = None  # Override in sub-classes

    # default directory where the training logs and models are saved
    LOG_ROOT_DIR = 'experiments'

    IMAGE_SIZE = (128, 64)

    IS_RGB = False

    NUM_CLASSES = 8 + 1  # number of classes

    # Number of training steps per epoch, when None, it will be calculated from data size
    STEPS_PER_EPOCH = None

    LEARNING_RATE = 0.0001
    LEARNING_MOMENTUM = 0.9

    NUM_EPOCH = 20
    BATCH_SIZE = 4
    VAL_BATCH_SIZE = 8

    # Used only when label maps needed to be scaled
    _LABEL_MULTIPLIER = 1.0

    # A function that takes a file name, and returns a subject id or a group id.
    # This is used while splittig data into train/val/test such that same group falls
    # on a same split. Set the value to None, if no grouping is needed
    image_grouping_function = [lambda x : x]

    IMG_DIR = [None]

    LABEL_DIR= [None]

    TRAIN_VAL_TEST_RATIOS = [(0.7, 0.1, 0.2)]
    USE_DATA_SPLIT_FILES= False
    DROPOUT_RATE = None
    VERBOSE = 1

    ## Augmentation parameters
    AUG_FLIP_HORZ = False
    AUG_FLIP_VERT = False

    # Rotation angle range x: wil be sampled from -x to x
    AUG_ROTATE = 0
    # When random resize is True, each forward pass receives different size images.
    # Therefore, the batch size is required to be 1
    RANDOM_RESIZE = False

    def __init__(self):

        assert self.NAME is not None, 'Name of the experiment is not set'
        if (not os.path.exists(self.LOG_ROOT_DIR)): os.mkdir(self.LOG_ROOT_DIR)
        self.MODEL_SAVE_DIR = os.path.join(self.LOG_ROOT_DIR, self.NAME)
        if (not os.path.exists(self.MODEL_SAVE_DIR)): os.mkdir(self.MODEL_SAVE_DIR)
        self.LOG_DIR = os.path.join(self.LOG_ROOT_DIR, self.NAME)
        if (self.RANDOM_RESIZE): assert self.BATCH_SIZE == 1, 'When random resize is enabled, batch size should be one'
        assert self.NUM_CLASSES > 1 and self.NUM_CLASSES < 256, 'Number of classes is out of range, should be in range of [2,255])'

        channels = 3 if self.IS_RGB else 1
        if (not self.RANDOM_RESIZE):
            self.INPUT_SHAPE = (self.IMAGE_SIZE[0], self.IMAGE_SIZE[1], channels)
        else:
            self.INPUT_SHAPE = (None, None, channels)

        assert self.IMG_DIR is not None and self.LABEL_DIR is not None, ' image or label diretory is not provided'



        checktype = lambda  field, atype:  type(getattr(self, field)) is atype

        assert checktype('IMG_DIR', list),' list expected'
        assert checktype('LABEL_DIR', list), ' list expected'
        assert checktype('IMAGE_EXT', list), ' list expected'
        assert checktype('image_grouping_function', list), ' list expected'
        print('Ratio:', self.TRAIN_VAL_TEST_RATIOS)
        assert checktype('TRAIN_VAL_TEST_RATIOS', list), ' list expected found ' + type(self.TRAIN_VAL_TEST_RATIOS)

        assert len(self.IMG_DIR) == len(self.LABEL_DIR), ' the number of image and label diretory  list should be same'
        assert len(self.IMG_DIR) == len(self.IMAGE_EXT), 'inconsistent size with IMG_DIR'
        assert len(self.IMG_DIR) == len(self.image_grouping_function), 'inconsistent size with IMG_DIR'

        assert len(self.IMG_DIR) == len(self.TRAIN_VAL_TEST_RATIOS), 'inconsistent size with IMG_DIR'






