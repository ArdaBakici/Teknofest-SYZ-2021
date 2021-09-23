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
from keras_unet_collection import losses
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model
import cv2
import hetorex
from hetorex import loss_functions
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
BATCH_SIZE = 2 # Highly dependent on d-gpu and system ram
STEPS_PER_EPOCH = 5949//BATCH_SIZE # 4646 IMPORTANT this value should be equal to file_amount/batch_size because we can't find file_amount from tf.Dataset you should note it yourself
VAL_STEPS_PER_EPOCH = 1274//BATCH_SIZE # 995 same as steps per epoch
MODEL_WEIGHTS_PATH = None#'./models/14_09-22_55/best.h5' # if not none model will be contiune training with these weights
# every shard is 200 files with 36 files on last shard
# Model Constants
BACKBONE = 'efficientnetb3'
# unlabelled 0, iskemik 1, hemorajik 2
CLASSES = ['iskemik', 'kanama']
LR = 0.0001
EPOCHS = 100
MODEL_SAVE_PATH = "./models"

date_name = datetime.now().strftime("%d_%m-%H_%M")
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
    aug = get_preprocessing(sm.get_preprocessing(BACKBONE))(image=image, mask=mask)
    image, mask = aug["image"].astype("float32"), aug["mask"].astype("float32")
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

n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

# ------ For debugging of tfrecords this is the old dataloader method ------
# define heavy augmentations
def get_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0)]
    
    return A.Compose(train_transform)


def get_preprocessing(preprocessing_fn):
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

# classes for data loading and preprocessing
class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['unlabelled', 'iskemik', 'kanama']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)
    
    
class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):    
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        # newer version of tf/keras want batch to be in tuple rather than list
        return tuple(batch)
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes) 

x_train_dir = './data/dataset1/train/'
y_train_dir = './data/dataset1/train_label/'
x_valid_dir = './data/dataset1/val/'
y_valid_dir = './data/dataset1/val_label/'
preprocess_input = sm.get_preprocessing(BACKBONE)

# Dataset for train images
train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    classes=CLASSES, 
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)

# Dataset for validation images
valid_dataset = Dataset(
    x_valid_dir, 
    y_valid_dir, 
    classes=CLASSES, 
    augmentation=None,
    preprocessing=get_preprocessing(preprocess_input),
)

train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

# check shapes for errors
assert train_dataloader[0][0].shape == (BATCH_SIZE, 512, 512, 3)
assert train_dataloader[0][1].shape == (BATCH_SIZE, 512, 512, 3)


# ------ End ------

#create model
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation, encoder_freeze=False)

# define optomizer
optim = keras.optimizers.Adam(LR)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
# TODO redefine class weights
dice_loss = sm.losses.DiceLoss(class_weights=np.array([2, 2, 0.2])) 
#multi_focal_tversky = losses.multiclass_focal_tversky(alpha=0.7, gamma=4/3)
focal_loss = sm.losses.CategoricalFocalLoss()
#total_loss = dice_loss + (1 * focal_tversky)
total_loss = dice_loss + (1 * focal_loss)
combo_loss = sm.losses.CategoricalFocalLoss()
#def hybrid_loss(y_true, y_pred):
#    loss_dice = dice_loss(y_true, y_pred)
#    loss_tversky = multi_focal_tversky(y_true, y_pred)
#    return loss_dice + loss_tversky

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

metrics = [sm.metrics.IOUScore(), sm.metrics.FScore()]

# compile keras model with defined optimozer, loss and metrics
model.compile(optim, total_loss, metrics)

if(MODEL_WEIGHTS_PATH is not None):
    model = load_model(MODEL_WEIGHTS_PATH, custom_objects={'dice_loss_plus_1focal_loss': total_loss,'iou_score': sm.metrics.IOUScore(), 'f1-score': sm.metrics.FScore()})
    for layer in model.layers[:]:
        layer.trainable = True

history = model.fit(
        train_dataloader, 
        steps_per_epoch=len(train_dataloader), 
        epochs=EPOCHS, 
        callbacks=callbacks, 
        validation_data=len(valid_dataloader), 
        validation_steps=VAL_STEPS_PER_EPOCH,
        #initial_epoch=5
    )

model_name = f'{history.history["val_iou_score"][-1]}iou_{datetime.now().strftime("%H_%M_%d_%m_%Y")}'
save_path = os.path.join(MODEL_SAVE_PATH, f"{model_name}.h5")
model.save(save_path)
