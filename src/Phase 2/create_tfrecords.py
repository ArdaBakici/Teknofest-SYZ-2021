import tensorflow as tf
import os 
import numpy as np
import cv2
import tqdm
import random

# Dataset Constants
SPLIT_DATASET = False
DATASET_SPLIT = ["train", "val", "test"]
DATASET_PATH = "./data/dataset1"
TRAIN_DIR = "train"
VAL_DIR = "validation"
TEST_DIR = "test"

DATA_DIR = "data"
LABEL_DIR = "label"

IMG_EXT = "png"

OUT_PATH = "./outdata/tfrecord/"

CLASS_VALUES = [1, 2]

MAX_FILES = 200

def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[serialize_array(value)])
    )

def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def parse_tfrecord_fn(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "path": tf.io.FixedLenFeature([], tf.string),
        "area": tf.io.FixedLenFeature([], tf.float32),
        "bbox": tf.io.VarLenFeature(tf.float32),
        "category_id": tf.io.FixedLenFeature([], tf.int64),
        "id": tf.io.FixedLenFeature([], tf.int64),
        "image_id": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
    example["bbox"] = tf.sparse.to_dense(example["bbox"])
    return example

# non keras
def serialize_array(array):
  array = tf.io.serialize_tensor(array).numpy()
  return array

def parse_single_image(image, label):
  
  #define the dictionary -- the structure -- of our single example
  data = {
        'image/height' : int64_feature(image.shape[0]),
        'image/width' : int64_feature(image.shape[1]),
        'image/depth' : int64_feature(image.shape[2]),
        'image/raw_image' : image_feature(image),
        'label/raw' : image_feature(label)
    }
  #create an Example, wrapping the single features
  out = tf.train.Example(features=tf.train.Features(feature=data))
  return out

def write_image_batches_to_tfr(img_path, label_path, filename:str="batch", max_files:int=100, out_dir:str="/data/tfrecord/"):
    img_filenames = tf.io.gfile.glob(f"./data/dataset1/data/*.png")
    print(img_filenames)
    random.shuffle(img_filenames)
    label_filenames = []
    for i in img_filenames:
        label_filenames.append(i.replace(img_path, label_path))
    # determine the number of shards (single TFRecord files) we need:
    assert len(img_filenames) == len(label_filenames)
    splits = (len(img_filenames)//max_files) + 1 #determine how many tfr shards are needed
    if len(img_filenames)%max_files == 0:
        splits-=1
    print(f"\nUsing {splits} shard(s) for {len(img_filenames)} files, with up to {max_files} samples per shard")
    os.makedirs(out_dir, exist_ok=True)
    file_count = 0
    for i in tqdm.tqdm(range(splits)):
        current_shard_name = f"{out_dir}tfrecord_{i+1}in{splits}_{filename}.tfrecords"
        writer = tf.io.TFRecordWriter(current_shard_name)

        current_shard_count = 0
        while current_shard_count < max_files: #as long as our shard is not full
            #get the index of the file that we want to parse now
            index = i*max_files+current_shard_count
            if index == len(img_filenames): #when we have consumed the whole data, preempt generation
                break
            
            #img = None
            #with open(img_filenames[index], 'rb') as file_reader:
            #    img = file_reader.read()
            img = cv2.imread(img_filenames[index])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(label_filenames[index], 0)
            masks = [(mask == v) for v in CLASS_VALUES]
            mask = np.stack(masks, axis=-1).astype('float')
            # add background if mask is not binary
            if mask.shape[-1] != 1:
                background = 1 - mask.sum(axis=-1, keepdims=True)
                mask = np.concatenate((mask, background), axis=-1)

            #create the required Example representation
            out = parse_single_image(image=img, label=mask)
            
            writer.write(out.SerializeToString())
            current_shard_count+=1
            file_count += 1
        writer.close()
    print(f"\nWrote {file_count} elements to TFRecord")

if __name__ == '__main__':
    if SPLIT_DATASET:
        for split in DATASET_SPLIT:
            print(f"Starting to process split **{split}**")
            split_img = os.path.join(DATASET_PATH, split)
            split_label = os.path.join(DATASET_PATH, f"{split}annot")
            write_image_batches_to_tfr(split_img, split_label, filename=split, max_files=MAX_FILES, out_dir=OUT_PATH)
    else:
        print(f"Starting the process.")
        split_img = os.path.join(DATASET_PATH, DATA_DIR)
        split_label = os.path.join(DATASET_PATH, LABEL_DIR)
        print(split_img)
        print(split_label)
        write_image_batches_to_tfr(split_img, split_label, filename="teknofest", max_files=MAX_FILES, out_dir=OUT_PATH)