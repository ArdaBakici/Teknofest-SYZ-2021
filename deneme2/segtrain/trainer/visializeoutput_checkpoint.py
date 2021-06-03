import os
os.environ['KERAS_BACKEND'] = "plaidml.keras.backend"
import colorsys
import cv2
import numpy as np
import os
from keras.callbacks import Callback
from dataflow import BatchData


class VisualizeOutputCheckpoint(Callback):
    def __init__(self, datasource, viz_dir, model=None, predict_batch_size=2, num_images=6):
        """

        :param test_datasource:
        :param viz_dir:
        :param model: if None, then the training model accessible through super class is used
        :param predict_batch_size:
        :param num_images:
        """

        super(VisualizeOutputCheckpoint, self).__init__()
        self.predict_batch_size = predict_batch_size

        if (not os.path.exists(viz_dir)):
            os.mkdir(viz_dir)

        self.viz_dir = viz_dir
        self.model_ = model
        num_images = np.min ([datasource.size(), num_images])
        self.val_imgs, self.val_labels_probmaps = next(BatchData(datasource, num_images).get_data())
        self.val_labels_probmaps = np.squeeze(self.val_labels_probmaps)
        self.num_images = num_images

    def on_batch_begin(self, batch, logs={}):
        pass

    def predict_(self, batch_size):

        model = self.model_ if self.model_ is not None else self.model

        pred = model.predict(self.val_imgs, batch_size=batch_size, verbose=False)
        if (type(pred) is list):
            pred_logits, softmax_out = pred
        else:
            softmax_out = pred
        softmax_out = np.asarray(softmax_out)

        return softmax_out

    def on_train_begin(self, logs={}):
        pass

    def on_train_end(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass




    def on_epoch_end(self, epoch, logs={}):
        softmax_out = self.predict_(batch_size=2)
        viz_out,_ = visualize_labels_overlay(softmax_out,self.val_imgs*255, self.num_images)
        viz_gt, val_imgs = visualize_labels_overlay(self.val_labels_probmaps, self.val_imgs*255, self.num_images)

        max_size=512
        scale  = max_size/np.max(viz_out.shape)
        if(scale<1):
           viz_out =  cv2.resize(viz_out,(0,0),fx=scale, fy=scale)
           viz_gt =  cv2.resize(viz_gt,(0,0),fx=scale, fy=scale)
           val_imgs = cv2.resize(val_imgs, (0,0), fx=scale, fy=scale)


        cv2.imwrite(os.path.join(self.viz_dir, 'epoch' + str(epoch) + '.jpeg'), viz_out)

        if(epoch ==0):
            cv2.imwrite(os.path.join(self.viz_dir, 'ground_truth_epoch' + str(epoch) + '.jpeg'), viz_gt)
            cv2.imwrite(os.path.join(self.viz_dir, 'val_images_epoch' + str(epoch) + '.jpeg'), val_imgs)









def  visualize_labels_overlay(prob_map, val_imgs, num_images, alpha_label=0.6):
    """

    :param prob_map: (n,H,W,C) where C= number of classes
    :param val_imgs: (n,H,W,C) where C=1 or 3
    :return:
    """

    n_classes = prob_map.shape[3]
    BG_INDEX = n_classes - 1
    label_maps = np.argmax(prob_map, axis=3)

    alphas = np.ones_like(label_maps) * alpha_label
    alphas[label_maps == BG_INDEX] = 0
    #alphas[label_maps != BG_INDEX] = 0.2
    alphas = np.expand_dims(alphas, -1)
    alphas = np.repeat(alphas,3,axis=3)
    viz = visualize_labelmaps(label_maps, N_class=prob_map.shape[3])

    if (val_imgs.shape[3] == 1):
        val_imgs = np.repeat(val_imgs, 3, axis=3)
    else:
        assert val_imgs.shape[3] == 3, 'channels not supported for visualization ' + str(val_imgs.shape[3])

    #alphas =0.5
    #alphas = alphas.astype(np.float32)
    final_viz = val_imgs * (1-alphas) +  (alphas) * viz
    a, b = get_factors(num_images)
    final_viz = stack_patches(final_viz, a, b)

    val_imgs = stack_patches(val_imgs, a,b)
    return final_viz, val_imgs





def visualize_labels_overlay_labelmap(label_maps, val_imgs, n_classes, alpha_label=0.6, BG_INDEX=None, stack_images=True):
    """

    :param label_maps: (n,H,W) where each pixel contain class index
    :param val_imgs: (n,H,W,C) where C=1 or 3
    :n_classes number of classes used in visualization. It should be consistent with label_maps
    :param BG_INDEX: index of background class, if None then n_classes - 1 will be used
    :return:
    """

    num_images = label_maps.shape[0]

    if (BG_INDEX is None):
        BG_INDEX = n_classes - 1

    alphas = np.ones_like(label_maps) * alpha_label
    alphas[label_maps == BG_INDEX] = 0
    # alphas[label_maps != BG_INDEX] = 0.2
    alphas = np.expand_dims(alphas, -1)
    alphas = np.repeat(alphas, 3, axis=3)
    viz = visualize_labelmaps(label_maps, N_class=n_classes)

    if (val_imgs.shape[3] == 1):
        val_imgs = np.repeat(val_imgs, 3, axis=3)
    else:
        assert val_imgs.shape[3] == 3, 'channels not supported for visualization ' + str(val_imgs.shape[3])

    # alphas =0.5
    # alphas = alphas.astype(np.float32)
    final_viz = val_imgs * (1 - alphas) + (alphas) * viz

    if(stack_images):
        a, b = get_factors(num_images)
        final_viz = stack_patches(final_viz, a, b)
        val_imgs = stack_patches(val_imgs, a,b)

    return final_viz, val_imgs


def visualize_labelmaps(label_maps, N_class, colors=None, background_index=None):
    """

    :param label_maps:
    :param N_class:
    :param colors:
    :param background_index:  is none, the last the background index is taken as N_class-1
    :return:
    """
    if (background_index is None):
        background_index = N_class - 1

    if (colors is None):
        colors = [colorsys.hsv_to_rgb(x * 1.0 / N_class, 0.6, 1) for x in range(N_class)]
    else:
        assert len(colors) == N_class

    colors[background_index] = (0, 0, 0)

    def get_labelmap_viz(alabel):
        viz = np.zeros((alabel.shape[0], alabel.shape[1], 3))
        for c in range(N_class):
            viz += _visualize_probmaps_oneclass(alabel, c, colors[c])
        return viz

    if (len(label_maps.shape) == 3):
        labels_viz = []
        for alabel in label_maps:
            viz = get_labelmap_viz(alabel)
            labels_viz.append(np.expand_dims(viz, 0))
        vizim = np.vstack(labels_viz)
        return vizim
    else:
        alabel = label_maps
        viz = get_labelmap_viz(alabel)
        return viz


def _visualize_probmaps_oneclass(alabel, class_label, color):
    alabel_r = np.zeros((alabel.shape[0], alabel.shape[1], 1))
    alabel_g = np.zeros((alabel.shape[0], alabel.shape[1], 1))
    alabel_b = np.zeros((alabel.shape[0], alabel.shape[1], 1))

    mask_class = alabel == class_label
    np.place(alabel_r, mask_class, color[0] * 255)
    np.place(alabel_g, mask_class, color[1] * 255)
    np.place(alabel_b, mask_class, color[2] * 255)

    im = np.concatenate([alabel_r, alabel_g, alabel_b], axis=2)
    return im.astype(np.uint8)


def _pad_patches_stack(patches, N):
    """
     Appends the image stack with the blank images to make total number of images in stack to be N
    :param patches:
    :param N:
    :return: padded patches
    """
    n_blank = N - patches.shape[0]

    # assert (n_blank <= ngrid_x and n_blank <= ngrid_y), ' too many blank grids, ' + str(n_blank)

    if n_blank > 0:
        ss = list(patches.shape)
        ss[0] = n_blank
        blank = np.zeros(tuple(ss))
        patches = np.vstack([patches, blank])

    return patches


def stack_patches(patches, nr_row, nr_col):
    """
    Stack patches, i.e,  convert the image patches to a single image
    :param patches:  N x H x W x c
    :return: nr_row*H x nr_col*W image
    """
    assert (patches.shape[0] <= nr_row * nr_col), 'The number of patches should be equal to nr_row*nr_col' + str(
        patches.shape[0]) + '<=' + str(nr_row) + 'x' + str(nr_col)

    n_blank = nr_col * nr_row - patches.shape[0]
    assert (n_blank <= nr_row and n_blank <= nr_col), ' too many blank grids,' + str(n_blank)

    patches = _pad_patches_stack(patches, nr_row * nr_col)

    rows = []
    for r in range(nr_row):
        cols = []
        for c in range(nr_col):
            # print r,c, r * nr_row + c
            cols.append(patches[r * nr_col + c, :, :, :])
        col = np.concatenate(cols, axis=1)
        rows.append(col)

    row = np.concatenate(rows, axis=0)
    return row


def get_factors(x):
    # This function takes a number x  and returns the two numbers a,b both factors of x
    # such  that abs(a-b) is minimum among all factors of x

    root= int(np.sqrt(x))
    if(root*root == int(x)):
        return root, root

    z = []
    for i in range(1, x + 1):
        if x % i == 0:
            z.append(i)
    n = int(len(z) / 2) - 1
    return z[n], z[n + 1]



if (__name__ == '__main__'):
    a = VisualizeOutputCheckpoint()
