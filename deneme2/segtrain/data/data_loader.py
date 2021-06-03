import os

import cv2
import dataflow as df
import numpy as np

from .data_directoryimages import DirectoryImagesLabeled
from .datautils import LabelMap2ProbabilityMap


def replace(match, target):
    def f(arr):
        ind_source = arr == match
        ind_target = arr == target
        arr[ind_source] = target
        arr[ind_target] = match
        return arr

    return f


def fix_channel(x):
    if (len(x.shape) == 2):
        x = np.expand_dims(x, -1)
    return x


class CropMultiple16(df.imgaug.ImageAugmentor):

    def __init__(self):
        super(CropMultiple16, self).__init__()
        self._init(locals())

    def get_transform(self, img):
        newh = img.shape[0] // 16 * 16
        neww = img.shape[1] // 16 * 16
        assert newh > 0 and neww > 0
        diffh = img.shape[0] - newh
        h0 = 0 if diffh == 0 else self.rng.randint(diffh)
        diffw = img.shape[1] - neww
        w0 = 0 if diffw == 0 else self.rng.randint(diffw)
        return df.imgaug.CropTransform(h0, w0, newh, neww)


def process_data(ds, num_classes, augment=True, IMAGE_SIZE=None, _label_multiplier=1.0,
                 aug_flip_horiz=True, aug_flip_vert=False, aug_rotate=0, random_resize=False):
    """
    Assumes that the background index in label map is the last one i.e.,  background_index=num_classes-1
    :param ds:
    :param num_classes:
    :param new_size:
    :param rotate_aug:
    :param crop_params:
    :return:
    """

    if (IMAGE_SIZE[0] is None): assert IMAGE_SIZE[0] == IMAGE_SIZE[1], 'if one dimension is none, other should be none'

    resizer = [df.imgaug.Resize(IMAGE_SIZE, interp=cv2.INTER_NEAREST)] if IMAGE_SIZE[0] else []

    augmentors_train_both = []

    if (random_resize):
        random_resizer = df.imgaug.RandomResize(xrange=(0.5, 1.4), yrange=(0.5, 1.4), aspect_ratio_thres=0.20,
                                                interp=cv2.INTER_NEAREST)
        augmentors_train_both.extend([random_resizer, CropMultiple16()])

    if (aug_rotate is not None and aug_rotate != 0): augmentors_train_both.append(df.imgaug.Rotation(aug_rotate))
    if (aug_flip_horiz): augmentors_train_both.append(df.imgaug.Flip(horiz=True, vert=False, prob=0.5))
    if (aug_flip_vert): augmentors_train_both.append(df.imgaug.Flip(horiz=False, vert=True, prob=0.5))

    augmentors_train_image = [df.imgaug.BrightnessScale([0.8, 1.2]),
                              df.imgaug.Contrast([0.8, 1.2])]

    augmentors_both = resizer
    augmentors_image = []
    if (augment):
        augmentors_both.extend(augmentors_train_both)
        augmentors_image.extend(augmentors_train_image)

    # to augment (make sure 0 is background)
    ds = df.MapDataComponent(ds, lambda x: x * _label_multiplier, index=1)
    # ds = df.MapDataComponent(ds, replace(num_classes-1,0), index=1)

    ds = df.AugmentImageComponents(ds, augmentors=augmentors_both, index=(0, 1))
    ds = df.AugmentImageComponent(ds, augmentors=augmentors_image)

    ds = df.MapDataComponent(ds, lambda x: x / 255.0, index=0)

    # convert any floating point during resize operation to uint8
    ds = df.MapDataComponent(ds, lambda x: x.astype(np.uint8), index=1)

    # ds = df.MapDataComponent(ds, replace(0,num_classes - 1), index=1)
    ds = LabelMap2ProbabilityMap(ds, label_map_index=1, num_classes=num_classes)

    ds = df.MapDataComponent(ds, fix_channel, index=0)

    ds = df.SelectComponent(ds, [0, 1])
    return ds


class SegDataLoader:
    def __init__(self, image_path, label_path, image_extension, grouping_function=None):
        self.data = DirectoryImagesLabeled(image_path=image_path,
                                           label_path=label_path,
                                           grouping_function=grouping_function,
                                           ext=image_extension)
        self._label_multiplier = 1.0
        self.image_path = image_path

    def get_dataflow_train_val_test(self, num_classes, ratios=(0.7, 0.2, 0.1), seed=0, isRGB=False, IMAGE_SIZE=None,
                                    aug_flip_horiz=True, aug_flip_vert=False, aug_rotate=0, random_resize=False):
        """
        Prepares the dataflow for given split configuration
        :param ratios:
        :param seed:
        :param isRGB:
        :param IMAGE_SIZE:
        :return:
        """
        assert (len(ratios) == 3) and np.sum(ratios) > np.finfo(float).eps - 1 and np.sum(ratios) < np.finfo(
            float).eps + 1, 'Expected three ratios which should sum to 1'
        [train, val, test], filenames = self.data.get_train_val_test_dataflow(seed=seed, split_info=ratios, isRGB=isRGB)
        self.filenames = filenames

        train = process_data(train, augment=True, IMAGE_SIZE=IMAGE_SIZE, num_classes=num_classes,
                             _label_multiplier=self._label_multiplier, aug_flip_horiz=aug_flip_horiz,
                             aug_flip_vert=aug_flip_vert, aug_rotate=aug_rotate, random_resize=random_resize)

        val = process_data(val, augment=False, IMAGE_SIZE=IMAGE_SIZE, num_classes=num_classes,
                           _label_multiplier=self._label_multiplier)
        test = process_data(test, augment=False, IMAGE_SIZE=IMAGE_SIZE, num_classes=num_classes,
                            _label_multiplier=self._label_multiplier)
        return train, val, test

    def split(self, config, ratios, seed=0):
        """
        Prepares the dataflow for given split configuration
        :param ratios:
        :param seed:
        :param isRGB:
        :param IMAGE_SIZE:
        :return:
        """

        num_classes = config.NUM_CLASSES
        # ratios = config.TRAIN_VAL_TEST_RATIOS
        isRGB = config.IS_RGB
        IMAGE_SIZE = config.IMAGE_SIZE
        aug_flip_vert = config.AUG_FLIP_VERT
        aug_flip_horiz = config.AUG_FLIP_HORZ
        aug_rotate = config.AUG_ROTATE
        random_resize = config.RANDOM_RESIZE

        if (config.USE_DATA_SPLIT_FILES):
            split_files = list(map(lambda x: os.path.join(self.image_path,x), ['train.txt', 'val.txt', 'test.txt']))
            assert(all(os.path.exists(f) for f in split_files)), 'missing one or more split files'
            split_info = split_files

        else:

            assert (len(ratios) == 3) and np.sum(ratios) > np.finfo(float).eps - 1 and np.sum(ratios) < np.finfo(
                float).eps + 1, 'Expected three ratios which should sum to 1'
            split_info=ratios

        [train, val, test], filenames = self.data.get_train_val_test_dataflow(seed=seed, split_info=split_info, isRGB=isRGB)
        self.filenames = filenames

        train = process_data(train, augment=True, IMAGE_SIZE=IMAGE_SIZE, num_classes=num_classes,
                             _label_multiplier=self._label_multiplier, aug_flip_horiz=aug_flip_horiz,
                             aug_flip_vert=aug_flip_vert, aug_rotate=aug_rotate, random_resize=random_resize)

        val = process_data(val, augment=False, IMAGE_SIZE=IMAGE_SIZE, num_classes=num_classes,
                           _label_multiplier=self._label_multiplier)
        test = process_data(test, augment=False, IMAGE_SIZE=IMAGE_SIZE, num_classes=num_classes,
                            _label_multiplier=self._label_multiplier)
        return train, val, test

    def get_filenames(self):
        [train_files, val_files, test_files] = self.filenames
        return [train_files, val_files, test_files]


if __name__ == '__main__':

    grouping_function = lambda x: x.split('-')[2]

    oct = SegDataLoader(image_path='../../data/images',label_path='../../data/labels', image_extension='.jpeg',
                        grouping_function=grouping_function)
    train, val, test = oct.get_dataflow_train_val_test(seed=0, ratios=(0.7, 0.2, 0.1))

    cc = 0
    unc_all = []
    for im, lab in train.get_data():
        print(im.shape, lab.shape)
