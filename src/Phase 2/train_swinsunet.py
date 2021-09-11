# Here is the imports
import os

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
FLAGS = ["tensorboard"] # tensorboard, mixed_precision
from tensorflow import keras
import numpy as np
import tensorflow as tf
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
import segmentation_models as sm
sm.set_framework("tf.keras")
if "mixed_precision" in FLAGS:
    print(f"PreTrain: Using Mixed Policy float16")
    keras.mixed_precision.set_global_policy('mixed_float16')
# keras.mixed_precision.set_global_policy('mixed_float16') normally this would provide extra speed for the model
# but in the case of 1660ti gpus they seem like they have tensor cores even they don't thus it slows down the model use this on higher powered models
from matplotlib import pyplot as plt
AUTOTUNE = tf.data.AUTOTUNE
import random
import albumentations as A
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard
from keras_unet_collection import models, losses
# segmentation_models could also use `tf.keras` if you do not have Keras installed
# or you could switch to other framework using `sm.set_framework('tf.keras')`

# Dataset Constants
DATASET_PATH = "./recordbase"
TRAIN_DIR = "train"
VAL_DIR = "val"
TEST_DIR = "test"

RECORD_ENCODING_TYPE = "ZLIB" # none if no encoding is used

# Pipeline parameters
BUFFER_SIZE = None # set buffer size to default value, change if you have bottleneck
SHUFFLE_SIZE = 256 # because dataset is too large huge shuffle sizes may cause problems with ram
BATCH_SIZE = 1 # Highly dependent on d-gpu and system ram
STEPS_PER_EPOCH = 1#5949//BATCH_SIZE # 4646 IMPORTANT this value should be equal to file_amount/batch_size because we can't find file_amount from tf.Dataset you should note it yourself
VAL_STEPS_PER_EPOCH = 1#1274//BATCH_SIZE # 995 same as steps per epoch
MODEL_WEIGHTS_PATH = None # if not none model will be contiune training with these weights
# every shard is 200 files with 36 files on last shard
# Model Constants
BACKBONE = 'efficientnetb3'
# unlabelled 0, iskemik 1, hemorajik 2
CLASSES = ['iskemik', 'kanama']
LR = 0.0001
EPOCHS = 20
MODEL_SAVE_PATH = "./models"

# Variables
train_dir = os.path.join(DATASET_PATH, TRAIN_DIR)
val_dir = os.path.join(DATASET_PATH, VAL_DIR)

train_filenames = tf.io.gfile.glob(f"{train_dir}/*.tfrecords")
val_filenames = tf.io.gfile.glob(f"{val_dir}/*.tfrecords")

random.shuffle(train_filenames) # shuffle tfrecord files order
random.shuffle(val_filenames)

# define callbacks for learning rate scheduling and best checkpoints saving
callbacks = [
    keras.callbacks.ModelCheckpoint(f'{MODEL_SAVE_PATH}/best_{datetime.now().strftime("%H_%M-%d_%m")}.h5', save_weights_only=False, save_best_only=True, mode='min'),
    keras.callbacks.ModelCheckpoint(f'{MODEL_SAVE_PATH}/epoch_{{epoch:02d}}_{datetime.now().strftime("%H_%M-%d_%m")}.h5', save_weights_only=False, save_freq=STEPS_PER_EPOCH*10, save_best_only=False, mode='min'),
    keras.callbacks.ReduceLROnPlateau(),
    keras.callbacks.CSVLogger(f'./customlogs/{datetime.now().strftime("%H_%M-%d_%m")}.csv')
]

if "tensorboard" in FLAGS:
    print(f"PreTrain: Using tensorboard")
    callbacks.append(
        TensorBoard(
            log_dir="logs",
            histogram_freq=0,
            write_graph=True,
            write_images=False,
            update_freq="epoch",
        ))

def aug_fn(image, mask):
    transforms = A.Compose([
            A.Rotate(limit=40),
            A.Flip(),
            ])
    aug_data = transforms(image=image, mask=mask)
    aug_img, aug_mask = aug_data["image"], aug_data["mask"]
    return aug_img, aug_mask

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)
    
def preprocessing_fn(image, mask):
    image = image/255.
    image = image.astype("float32")
    return image, mask

def parse_examples_batch(examples):
    feature_description = {
        'image/raw_image' : tf.io.FixedLenFeature([], tf.string),
        'label/raw' : tf.io.FixedLenFeature([], tf.string)
    }
    samples = tf.io.parse_example(examples, feature_description)
    return samples

def prepare_sample(features):
    image = tf.vectorized_map(lambda x: tf.io.parse_tensor(x, out_type = tf.uint8), features["image/raw_image"])
    label = tf.vectorized_map(lambda x: tf.io.parse_tensor(x, out_type = tf.float32), features["label/raw"])
    image, label = tf.vectorized_map(lambda x: tf.numpy_function(func=preprocessing_fn, inp=x, Tout=(tf.float32, tf.float32)), [image, label])
    return image, label

def prepare_sample_aug(features):
    image = tf.vectorized_map(lambda x: tf.io.parse_tensor(x, out_type = tf.uint8), features["image/raw_image"])
    label = tf.vectorized_map(lambda x: tf.io.parse_tensor(x, out_type = tf.float32), features["label/raw"]) # this was float64
    image, label = tf.vectorized_map(lambda x: tf.numpy_function(func=aug_fn, inp=x, Tout=(tf.uint8, tf.float32)), [image, label])
    image, label = tf.vectorized_map(lambda x: tf.numpy_function(func=preprocessing_fn, inp=x, Tout=(tf.float32, tf.float32)), [image, label])
    return image, label

def get_dataset_optimized(filenames, batch_size, shuffle_size, augment=True):
    record_dataset = tf.data.TFRecordDataset(filenames, compression_type=RECORD_ENCODING_TYPE, num_parallel_reads=AUTOTUNE)
    if shuffle_size > 0:
        record_dataset = record_dataset.shuffle(shuffle_size)
    record_dataset = (record_dataset
                    .repeat(EPOCHS)
                    .batch(batch_size=batch_size)
                    .map(map_func=parse_examples_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE))
    if augment:
        record_dataset = record_dataset.map(map_func=prepare_sample_aug, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        record_dataset = record_dataset.map(map_func=prepare_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return record_dataset.prefetch(tf.data.experimental.AUTOTUNE)

model = models.swin_unet_2d((512, 512, 3), filter_num_begin=64, n_labels=3, depth=3, stack_num_down=2, stack_num_up=2, 
                            patch_size=(2, 2), num_heads=[4, 8, 8], window_size=[4, 2, 2, 2], num_mlp=512, 
                            output_activation='Softmax', shift_window=True, name='swin_unet')

optim = keras.optimizers.Adam(LR)

dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.45, 0.45, 0.1])) 
focal_tversky = losses.focal_tversky
focal_loss = sm.losses.CategoricalFocalLoss()
#total_loss = dice_loss + (1 * focal_tversky)
total_loss = dice_loss + (1 * focal_loss)

def hybrid_loss(y_true, y_pred):
    loss_dice = dice_loss(y_true, y_pred)
    loss_tversky = focal_tversky(y_true, y_pred)
    print(f"loss_dice is {loss_dice}")
    print(f"loss_tverksy is {loss_tversky}")
    return loss_dice + loss_tversky

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

metrics = [sm.metrics.IOUScore(), sm.metrics.FScore()]

# compile keras model with defined optimozer, loss and metrics
model.compile(optim, tota, metrics)

history = model.fit(
        get_dataset_optimized(train_filenames, BATCH_SIZE, SHUFFLE_SIZE), 
        steps_per_epoch=STEPS_PER_EPOCH, 
        epochs=EPOCHS, 
        callbacks=callbacks, 
        validation_data=get_dataset_optimized(val_filenames, BATCH_SIZE, 0), 
        validation_steps=VAL_STEPS_PER_EPOCH,
    )

model_name = f'{history.history["val_iou_score"][-1]}iou_{datetime.now().strftime("%H_%M-%d_%m_%Y")}'
save_path = f"MODEL_SAVE_PATH/{model_name}.h5"
model.save(save_path)