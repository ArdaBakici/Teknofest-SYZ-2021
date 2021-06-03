import os
os.environ['KERAS_BACKEND'] = "plaidml.keras.backend"
import keras
import numpy as np
from dataflow import RandomMixData, ConcatData, BatchData
from keras.callbacks import CSVLogger

from segtrain.data.datautils import write_text
from segtrain.data.data_loader import SegDataLoader
from segtrain.models.unet import create_unet_model
from segtrain.trainer.trainer import KerasTrainer
from segtrain.trainer.visializeoutput_checkpoint import VisualizeOutputCheckpoint



def get_data_source(config):
    train = [];
    val = [];
    test = []
    train_files = [];
    val_files = [];
    test_files = []

    for imgdir, labeldir, ext, grouper in zip(config.IMG_DIR, config.LABEL_DIR, config.IMAGE_EXT,
                                              config.image_grouping_function):
        dloader = SegDataLoader(image_path=imgdir, label_path=labeldir, image_extension=ext,
                                grouping_function=grouper)
        dloader._label_multiplier = config._LABEL_MULTIPLIER


        splits = config.TRAIN_VAL_TEST_RATIOS[0]

        train_, val_, test_ = dloader.split(config, ratios=splits, seed=0)
        train.append(train_)
        val.append(val_)
        test.append(test_)

        [train_files_, val_files_, test_files_] = dloader.get_filenames()
        train_files.extend(train_files_)
        val_files.extend(val_files_)
        test_files.extend(test_files_)

    train = RandomMixData(train) if len(train) >1 else train[0]
    val = ConcatData(val) if len(val) >1 else val[0]
    test = ConcatData(test) if len(test) >1 else test[0]

    print('Data stats, train:', train.size(), ' val:', val.size(), ' test:', test.size())

    return train, val, test, [train_files, val_files, test_files]


def train_segmentation_network(config, models=None):
    """
    Train the segmentation model fo a given configuration
    :param config: configuration class
    :param models: the user provided model. If None, a default unet model is trained
    :return:
    """
    INPUT_SHAPE = config.INPUT_SHAPE

    train, val, test, [train_files, val_files, test_files] = get_data_source(config)

    exp_dir = os.path.join(config.LOG_ROOT_DIR, config.NAME)
    # save image names for each set
    np.savetxt(os.path.join(exp_dir, 'train_images.csv'), np.asarray(train_files), delimiter=',', fmt='%s')
    np.savetxt(os.path.join(exp_dir, 'val_images.csv'), np.asarray(val_files), delimiter=',', fmt='%s')
    np.savetxt(os.path.join(exp_dir, 'test_images.csv'), np.asarray(test_files), delimiter=',', fmt='%s')

    if (models is None):
        print('Creating Unet model..')
        models = create_unet_model(N_classes=config.NUM_CLASSES, input_shape=INPUT_SHAPE,
                                   dropout_rate=config.DROPOUT_RATE, learning_rate=config.LEARNING_RATE)

    # val = df.FixedSizeData(val, 5)
    trainer = KerasTrainer(train_ds=train, model=models, prefix=config.NAME, model_save_dir=config.MODEL_SAVE_DIR,
                           val_ds=val)

    viz_dir = os.path.join(config.LOG_ROOT_DIR, config.NAME, 'viz')
    if (not os.path.exists(viz_dir)): os.mkdir(viz_dir)

    chkpt_predictions = VisualizeOutputCheckpoint(datasource=val, model=models.model_main, viz_dir=viz_dir)
    logger = CSVLogger(filename=os.path.join(exp_dir, 'train_logs.csv'))

    callbacks = [chkpt_predictions, logger]
    trainer.train(batch_size=config.BATCH_SIZE, val_batch_size=1, num_epochs=config.NUM_EPOCH,
                  steps_per_epoch=config.STEPS_PER_EPOCH, verbose=config.VERBOSE, additional_callbacks=callbacks, prefetch_data=False)

    return models.model_main
