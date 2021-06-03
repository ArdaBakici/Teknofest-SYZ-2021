import os
os.environ['KERAS_BACKEND'] = "plaidml.keras.backend"
import keras

import keras.backend as K
from dataflow import (
    BatchData, RepeatedData, MultiProcessRunnerZMQ)
from keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.callbacks import LearningRateScheduler

from .modelcheckpoint import CustomModelCheckpointCallback

def get_interval_lrscheduler_callback(model, epoch_interval, factor):
    def scheduler(epoch):
        if epoch % epoch_interval == 0 and epoch != 0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr * factor)
            print("lr changed to {}".format(lr * factor))
        return K.get_value(model.optimizer.lr)

    lr_decay = LearningRateScheduler(scheduler)
    return lr_decay


def model_saver_callback(trainer, epoch_interval):
    class ModelSaver(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (epoch % epoch_interval == 0 and epoch != 0):
                trainer.save_model_hd5('model_snapshot' + str(epoch))

    return ModelSaver()


class LearningRatePrinter(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        _lr = K.eval(self.model.optimizer.lr)
        print('lr:', _lr)


class KerasTrainer:
    def __init__(self, train_ds, model, prefix, model_save_dir, val_ds=None):
        self.train_ds = train_ds
        self.model = model
        self.prefix = prefix
        self.model_save_dir = model_save_dir
        self.val_ds = val_ds
        self.train_losses = []
        self.val_losses = []
        self.val_eval_epochs = []

    def train(self, batch_size, num_epochs, steps_per_epoch=None, lr_decay_type='plateau', init_learn_rate=None,
              verbose=0, data_grouper=None, additional_callbacks=[], val_batch_size=64, hook_tensorbord=False,
              chkpt_monitor=('val_loss', 'auto'), prefetch_data = False):
        """

        :param batch_size:
        :param num_epochs:
        :param steps_per_epoch:
        :param lr_decay_type: 'interval' or 'plateau'
        :return:
        """

        assert lr_decay_type == 'plateau' or lr_decay_type == 'interval', 'invalid option for lr_decay_type'

        ds_train_ = BatchData(self.train_ds, batch_size, remainder=False)
        if (steps_per_epoch is None):
            steps_per_epoch = ds_train_.size()

        if (data_grouper is not None):
            ds_train_ = data_grouper(ds_train_)

        #for parallel loading
        if(prefetch_data): ds_train_ = MultiProcessRunnerZMQ(ds_train_, num_proc=15)
        #ds_train_ = BatchData(ds_train_, 256)


        ds_train_ = RepeatedData(ds_train_, -1)

        ds_train_.reset_state()
        batcher_train = ds_train_.get_data()

        ds_val_ = BatchData(self.val_ds, val_batch_size, remainder=True)
        if (data_grouper is not None):
            ds_val_ = data_grouper(ds_val_)

        # ds_val_ = FixedSizeData(ds_val_ , ds_val_.size()/1) #only evaluate on the first 50% of the data
        val_steps = ds_val_.size()
        ds_val_ = RepeatedData(ds_val_, -1)

        ds_val_.reset_state()
        batcher_val = ds_val_.get_data()
        # val_steps =  20#ds_val_.size()/2  # only evaluate on 50% of data

        if (init_learn_rate is not None):
            K.set_value(self.model.model_train.optimizer.lr, init_learn_rate)

        print("Training with:  ")
        print('    nepochs', num_epochs)
        print('    number of iterations/epoch', steps_per_epoch)

        print('lr before for loop', K.get_value(self.model.model_train.optimizer.lr))

        if (lr_decay_type == 'plateau'):
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='auto', factor=0.25, patience=5, min_lr=1e-6)
        else:
            reduce_lr = get_interval_lrscheduler_callback(self.model.model_train, epoch_interval=18, factor=0.1)

        if (not os.path.exists(self.model_save_dir)):
            os.mkdir(self.model_save_dir)

        model_filepath = os.path.join(self.model_save_dir, self.prefix + '.hd5')

        monitor, mode = chkpt_monitor
        if (self.model.multigpu_train):
            model_checkpoint = CustomModelCheckpointCallback(model_filepath, self.model.model_main, monitor=monitor,
                                                             verbose=1, save_best_only=True,
                                                             save_weights_only=True, mode=mode, period=1)
        else:
            model_checkpoint = ModelCheckpoint(model_filepath, monitor=monitor, verbose=1, save_best_only=True,
                                               save_weights_only=True, mode=mode, period=1)

        lr_printer = LearningRatePrinter()

        callbacks = [reduce_lr, lr_printer, model_checkpoint]

        self.save_model_json()

        def new_batcher(b):
            for d in b:
                yield  tuple(d)#(d[0], d[1])

        self.model.model_train.fit_generator(new_batcher(batcher_train), steps_per_epoch=steps_per_epoch, epochs=num_epochs,
                                       verbose=verbose,
                                       callbacks=callbacks, validation_data=new_batcher(batcher_val),
                                       validation_steps=val_steps)

    def save_model_json(self):

        print("Saving model  structure as json")
        if (not os.path.exists(self.model_save_dir)):
            os.mkdir(self.model_save_dir)

        with open(os.path.join(self.model_save_dir, self.prefix + ".json"), "w") as text_file:
            text_file.write(self.model.model_main.to_json())
        print("Done")

    def save_model_hd5(self, filename_prefix):

        print("Saving model  structure as json")
        if (not os.path.exists(self.model_save_dir)):
            os.mkdir(self.model_save_dir)

        self.model.model_main.save(os.path.join(self.model_save_dir, filename_prefix + '.hd5'))
