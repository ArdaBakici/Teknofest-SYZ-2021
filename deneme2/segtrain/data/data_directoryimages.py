import glob
import os
import random

import cv2
import numpy as np
from dataflow import RNGDataFlow

from .datautils import Splitrandom


class DirectoryImagesCommon:

    def __init__(self, image_path, label_path, ext='*.png', grouping_function=None):

        self.files = glob.glob(image_path + '/*' + ext)
        self.label_path = label_path
        self.image_path = image_path

        if (len(self.files) == 0):
            raise Exception('No files found!', image_path)

        data = []
        for f in self.files:
            image_name = os.path.basename(f)
            group_id = grouping_function(image_name) if grouping_function is not None else image_name
            label_file = os.path.join(self.label_path, image_name.split('.')[0] + '.png')
            data.append((f, label_file, group_id, image_name))

        self.data = data


class DirectoryImagesTest(DirectoryImagesCommon):

    def __init__(self, image_path, ext='*.png', isRGB=False):
        grouping_function_fake = lambda x: x[0]
        self.isRGB = isRGB
        super().__init__(image_path, label_path='', ext=ext,
                         grouping_function=grouping_function_fake)

    def get_all_dataflow(self):
        return SegmentationData(self.data, loadLabels=False, isRGB=self.isRGB)




def read_text(in_file):
    lines=''
    with open(in_file) as f:
        lines = f.readlines()
        lines = list(map(lambda x: x.replace('\n', ''), lines))
    return lines


class DirectoryImagesLabeled(DirectoryImagesCommon):

    def __init__(self, image_path, label_path, ext='.png', grouping_function=None):
        super().__init__(image_path, label_path, ext, grouping_function)

    def get_file_lists(self, data_splits):
        extract_filename = lambda x: [i[0].split('/')[-1] for i in list(x)]
        files = [ extract_filename(s) for s in data_splits]
        return  files


    def use_predefined_split(self, splits_files):
        """
        Use splits defined by image names in a  text file
        :param splits:
        :return:
        """
        data_dic = {x[3]: x for x in self.data}
        splits=[]
        for sp_file in splits_files:
            names = read_text(sp_file)
            asplit = [data_dic[x] for  x in names]
            splits.append(asplit)
        return splits



    def get_train_val_test_dataflow(self, seed, split_info=(0.7, 0.2, 0.1), isRGB=False):
        """

        :param seed: seed for random splitting
        :param split_info: ratio of splits, should sum to 1
        :param isRGB: whether the image to load in RGB format
        :return: a tuple: first array of maskdataflow and second an array of filenames
        """
        if(type(split_info[0]) is np.float):
            same_patient = lambda sample: sample[2]
            splits = Splitrandom(ratios=split_info, seed=seed, group_func=same_patient)(self.data)
        elif type(split_info[0] is str):
            splits = self.use_predefined_split(split_info)

        mask_data_splits = [SegmentationData(split, isRGB=isRGB) for split in splits]

        file_list = self.get_file_lists(splits)

        return mask_data_splits, file_list

    def get_all_dataflow(self):
        return SegmentationData(self.data)


class SegmentationData(RNGDataFlow):
    """
    Dataflow for the segmentation data with images, and pixelwise labels. Each label image have value from
    0 to number_of_classes-1.
    """

    def __init__(self, data, isRGB=False, loadLabels=True, shuffle=False):
        """

        :param data:
        :param isRGB:
        :param loadLabels:
        :param shuffle:
        """
        self.path_data = data
        self.isRGB = isRGB
        self.loadLabels = loadLabels
        self.shuffle = shuffle
        self.rng = random.Random()

    def size(self):
        return len(self.path_data)

    def get_data(self):
        idxs = np.arange(self.size())
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:

            image_path, label_path, group_id, image_name = self.path_data[k]

            if (not self.isRGB):
                im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            else:
                im = cv2.imread(image_path)

            assert im is not None, 'image not loaded ' + image_path

            if (self.loadLabels):
                label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                assert label is not None, 'Labels not loaded ' + label_path
            else:
                label = np.zeros(np.shape(im))

            yield [im, label, image_name]


if __name__ == '__main__':

    patient_id_extractor_function = lambda x: x.split('-')[2]
    oct = DirectoryImagesLabeled(image_path='gt_data/images',
                                 label_path='gt_data/labels',
                                 grouping_function=patient_id_extractor_function)
    train, val, test = oct.get_train_val_test_dataflow(seed=0, split_info=(0.7, 0.2, 0.1))

    cc = 0
    unc_all = []
    for im, lab, pid in test.get_data():
        print(im.shape, lab.shape, pid)
