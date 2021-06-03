from segtrain.train_model import train_segmentation_network
from segtrain.eval_model import evaluate_segmentation_network, batch_predict
from segtrain.trainer.Config import Config


class MNISTConfig(Config):
    """Configuration for training on the demo mnist segmentation  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name unique to an experiment.
    NAME = "unet_1"

    # the directory where the output of the experiments will be saved.
    LOG_ROOT_DIR = 'temp'

    # size of input image as (HEIGHT, WIDTH) format. To enable dynamic size use (None, None)
    IMAGE_SIZE = (512, 512)
    # extension of the input image
    IMAGE_EXT = ['.png']

    # Flag denoting if the input image should be loaded in BGR format. If False, the image will be loaded as graysale
    IS_RGB = False

    # Number of classes
    NUM_CLASSES = 3  # 10 digits + background

    # Number of training steps per epoch, when None, it will be calculated from data size
    STEPS_PER_EPOCH = None

    LEARNING_RATE = 0.0001
    LEARNING_MOMENTUM = 0.9

    # Number of training epochs
    NUM_EPOCH = 5

    # Training mini-batch size. If IMAGE_SIZE is set to (None, None) then BATCH_SIZE should be 1
    BATCH_SIZE = 1

    # Validation batch size
    VAL_BATCH_SIZE = 4

    # A function that takes a file name, and returns a subject id or a group id. This is used while splittig data into
    # train/val/test such that same group falls on a same split. Set the value to staticmethod(None), if no grouping is needed

    image_grouping_function = [None]#[lambda x: x.split('_')[0]]

    # directory where training images are saved.
    IMG_DIR = ['data/images']

    # directory where training label images are saved.
    # The label file should be in .png format and have same name as images in IMG_DIR folder.
    LABEL_DIR = ['data/labels']

    TRAIN_VAL_TEST_RATIOS = [(0.2, 0.3, 0.3)]
    # Flag denoting if the data split files are to be used. Following split files will be searched
    # in IMG_DIR, 'train.txt', 'val.txt', 'test.txt'.
    USE_DATA_SPLIT_FILES = False

    DROPOUT_RATE = 0.2
    VERBOSE = 1

    ##Augmentation parameters
    AUG_FLIP_HORZ = False

    AUG_FLIP_VERT = False

    # Rotation angle range in degrees x: wil be sampled from -x to x
    AUG_ROTATE = 5
    # When random resize option is enabled, batch size should be 1
    RANDOM_RESIZE = False


if __name__ == '__main__':

    config = MNISTConfig()
    train_segmentation_network(config)
    evaluate_segmentation_network(config)

    #predict on any images in a given model
    #batch_predict('mnist_data/temp', 'mnist_data/temp_out', config=config,image_extension='.jpeg')
