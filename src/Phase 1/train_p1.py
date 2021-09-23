# Here is the imports
import os
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
FLAGS = ["no_full_train"] # tensorboard, mixed_precision, no_pretrain, no_finetune, qubvel, no_full_train
from tensorflow import keras
import numpy as np
import tensorflow as tf
if "mixed_precision" in FLAGS:
    print(f"PreTrain: Using Mixed Policy float16")
    keras.mixed_precision.set_global_policy('mixed_float16')
AUTOTUNE = tf.data.AUTOTUNE
import random
import albumentations as A
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.applications import EfficientNetB4
import efficientnet.tfkeras as eff
from tensorflow.keras.utils import plot_model

DATASET_PATH = "./final_recordbase"
TRAIN_DIR = "train"
VAL_DIR = "val"
TEST_DIR = "test"

RECORD_ENCODING_TYPE = "ZLIB" # none if no encoding is used

# Pipeline parameters
BUFFER_SIZE = None # set buffer size to default value, change if you have bottleneck
SHUFFLE_SIZE = 256 # because dataset is too large huge shuffle sizes may cause problems with ram
BATCH_SIZE = 16 # Highly dependent on d-gpu and system ram
STEPS_PER_EPOCH = 6486//BATCH_SIZE # 4977 IMPORTANT this value should be equal to file_amount/batch_size because we can't find file_amount from tf.Dataset you should note it yourself
VAL_STEPS_PER_EPOCH = 150//BATCH_SIZE # 1659 same as steps per epoch
MODEL_WEIGHTS_PATH = None #'./models/21_09-22_51-eff_2_stage/best.h5' # if not none model will be contiune training with these weights

# inme_yok, inme_var
CLASSES = ['inme_yok', 'inme_var']
LR = 0.0001
IMG_SIZE = 512
FIRST_EPOCHS = 10
FINE_TUNE_EPOCHS = FIRST_EPOCHS + 30
FULLY_TRAIN_EPOCHS = FINE_TUNE_EPOCHS + 30 
MODEL_SAVE_PATH = "./models"

specifier_name = 'eff_final_recordbase'
date_name = f'{datetime.now().strftime("%d_%m-%H_%M")}-{specifier_name}'

# Variables
train_dir = os.path.join(DATASET_PATH, TRAIN_DIR)
val_dir = os.path.join(DATASET_PATH, VAL_DIR)

train_filenames = tf.io.gfile.glob(f"{train_dir}/*.tfrecords")
val_filenames = tf.io.gfile.glob(f"{val_dir}/*.tfrecords")

random.shuffle(train_filenames) # shuffle tfrecord files order
random.shuffle(val_filenames)

os.makedirs(f'{MODEL_SAVE_PATH}/{date_name}', exist_ok=True)

# define callbacks for learning rate scheduling and best checkpoints saving
callbacks = [
    keras.callbacks.ModelCheckpoint(f'{MODEL_SAVE_PATH}/{date_name}/best.h5', save_weights_only=False, save_best_only=True, mode='min'),
    keras.callbacks.ModelCheckpoint(f'{MODEL_SAVE_PATH}/{date_name}/epoch_{{epoch:02d}}.h5', save_weights_only=False, save_freq=STEPS_PER_EPOCH*10, save_best_only=False, mode='min'),
    #keras.callbacks.ModelCheckpoint(f'{MODEL_SAVE_PATH}/{date_name}/weights_{{epoch:02d}}.h5', save_weights_only=True, save_freq=STEPS_PER_EPOCH*5, save_best_only=False, mode='min'),
    keras.callbacks.ReduceLROnPlateau(),
    keras.callbacks.CSVLogger(f'./customlogs/{date_name}.csv')
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

def aug_fn(image):
    transforms = A.Compose([
            A.Rotate(limit=40),
            A.Flip(),
            ])
    aug_data = transforms(image=image)
    aug_img = aug_data["image"]
    return aug_img

def preprocessing_fn(image):
    image = eff.preprocess_input(image)
    return image 

def normal_preprocess(image):
    image = image.astype("float32")
    return image 

def parse_examples_batch(examples):
    feature_description = {
        'image' : tf.io.FixedLenFeature([], tf.string),
        'label' : tf.io.FixedLenFeature([], tf.float32)
    }
    samples = tf.io.parse_example(examples, feature_description)
    return samples

def prepare_sample(features):
    image = tf.vectorized_map(lambda x: tf.io.parse_tensor(x, out_type = tf.uint8), features["image"])
    label = features["label"]
    if "qubvel" in FLAGS:
        image = tf.vectorized_map(lambda x: tf.numpy_function(func=preprocessing_fn, inp=x, Tout=(tf.float32)), [image])
    else:
        print("No qubvel")
        image = tf.vectorized_map(lambda x: tf.numpy_function(func=normal_preprocess, inp=x, Tout=(tf.float32)), [image])
    return image, label

def prepare_sample_aug(features):
    image = tf.vectorized_map(lambda x: tf.io.parse_tensor(x, out_type = tf.uint8), features["image"])
    label = features["label"]
    image = tf.vectorized_map(lambda x: tf.numpy_function(func=aug_fn, inp=x, Tout=(tf.uint8)), [image])
    image = tf.vectorized_map(lambda x: tf.numpy_function(func=preprocessing_fn, inp=x, Tout=(tf.float32)), [image])
    return image, label

def get_dataset_optimized(filenames, batch_size, epoch_num, shuffle_size, augment=True):
    record_dataset = tf.data.TFRecordDataset(filenames, compression_type=RECORD_ENCODING_TYPE, num_parallel_reads=AUTOTUNE)
    if shuffle_size > 0:
        record_dataset = record_dataset.shuffle(shuffle_size)
    record_dataset = (record_dataset
                    .repeat(epoch_num)
                    .batch(batch_size=batch_size)
                    .map(map_func=parse_examples_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE))
    if augment:
        record_dataset = record_dataset.map(map_func=prepare_sample_aug, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        record_dataset = record_dataset.map(map_func=prepare_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return record_dataset.prefetch(tf.data.experimental.AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
])

inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = data_augmentation(inputs)


if "qubvel" in FLAGS:
    model = eff.EfficientNetB4(
        include_top=False,
        weights="noisy-student", # imagenet, noisy-student 
        #input_tensor=inputs,
        input_shape=(512, 512, 3)
    )
    
else:    
    model = EfficientNetB4(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
        input_shape=(512, 512, 3),
    )

model.trainable = False

# Rebuild top
x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
x = layers.BatchNormalization()(x)
# possible 20, 31 - block71 expand_conv, block6a_project_conv 
top_dropout_rate = 0.2
x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
x = layers.Dense(1, name="pred")(x)
outputs = layers.Activation(activation='sigmoid', dtype='float32', name="result_activ")(x)

if "qubvel" in FLAGS:
    model = keras.models.Model(model.input, outputs, name="EfficientNet")
else:
    model = keras.models.Model(inputs, outputs, name="EfficientNet")

optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

model.compile(
    optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=["accuracy"]
)


if(MODEL_WEIGHTS_PATH is not None):
    model = load_model(MODEL_WEIGHTS_PATH)
    for layer in model.layers:
        layer.trainable = False


if not "no_pretrain" in FLAGS:
    history = model.fit(
            get_dataset_optimized(train_filenames, BATCH_SIZE, FIRST_EPOCHS, SHUFFLE_SIZE, augment=False), 
            steps_per_epoch=STEPS_PER_EPOCH, 
            epochs=FIRST_EPOCHS, 
            callbacks=callbacks, 
            validation_data=get_dataset_optimized(val_filenames, BATCH_SIZE, FIRST_EPOCHS, 0, augment=False), 
            validation_steps=VAL_STEPS_PER_EPOCH,
            #initial_epoch=5
        )

if not "no_finetune" in FLAGS:
    set_trainable = False
    for layer in model.layers:
        if layer.name == 'block6a_project_conv':
            set_trainable = True
        if set_trainable:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
        else:
            layer.trainable = False

    #for layer in model.layers[-20:]:
    #    if not isinstance(layer, layers.BatchNormalization):
    #        layer.trainable = True
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    model.compile(
    optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=["accuracy"]
    )

    history = model.fit(
            get_dataset_optimized(train_filenames, BATCH_SIZE, FINE_TUNE_EPOCHS, SHUFFLE_SIZE, augment=False), 
            steps_per_epoch=STEPS_PER_EPOCH, 
            epochs=FINE_TUNE_EPOCHS, 
            callbacks=callbacks, 
            validation_data=get_dataset_optimized(val_filenames, BATCH_SIZE, FINE_TUNE_EPOCHS, 0, augment=False), 
            validation_steps=VAL_STEPS_PER_EPOCH,
            initial_epoch=FIRST_EPOCHS
        )

if not "no_full_train" in FLAGS:
    set_trainable = False
    for layer in model.layers:
        if layer.name == 'block6a_project_conv':
            set_trainable = True
        if set_trainable:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
        else:
            layer.trainable = False

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    model.compile(
    optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=["accuracy"])
    history = model.fit(
            get_dataset_optimized(train_filenames, BATCH_SIZE, FULLY_TRAIN_EPOCHS, SHUFFLE_SIZE, augment=True), 
            steps_per_epoch=STEPS_PER_EPOCH, 
            epochs=FULLY_TRAIN_EPOCHS, 
            callbacks=callbacks, 
            validation_data=get_dataset_optimized(val_filenames, BATCH_SIZE, FULLY_TRAIN_EPOCHS, 0, augment=False), 
            validation_steps=VAL_STEPS_PER_EPOCH,
            initial_epoch=FINE_TUNE_EPOCHS
        )

model.save(f'{MODEL_SAVE_PATH}/{date_name}/final.h5')