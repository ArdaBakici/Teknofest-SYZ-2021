import cv2
import dataflow as df
import dataflow as df
import numpy as np
import os

from segtrain.data.data_directoryimages import DirectoryImagesTest, SegmentationData
from segtrain.models.utils import load_tfkeras_model
from segtrain.train_model import get_data_source
from segtrain.trainer.visializeoutput_checkpoint import visualize_labels_overlay_labelmap
from segtrain.data.datautils import  write_text


def evaluate_on(model, datasource):
    images, gt = next(df.BatchData(datasource, datasource.size()).get_data())
    gt = np.squeeze(gt)
    out = model.predict(images, verbose=True)
    _, dices, verbose = compute_dice_metric(preds=out, labels=gt)

    return verbose, dices


def evaluate_segmentation_network(config, model=None, custom_objects={}):
    """
    Evaluates model by computing dice metric between ground truth and predictions on validation and test set
    :param config:
    :param model:
    :param custom_objects:
    :return:
    """
    if (model is None):
        model = load_tfkeras_model(config.MODEL_SAVE_DIR, file_name_prefix=config.NAME, model=model,
                                   custom_objects=custom_objects)
    _, val, test, [train_files, val_files, test_files] = get_data_source(config)
    val_results, dices_val = evaluate_on(model, val)
    test_results, dices_test = evaluate_on(model, test)
    print('Validation results: ', val_results)
    print('Test results: ', test_results)
    write_text(os.path.join(config.LOG_DIR, 'val_results.txt'), val_results)
    write_text(os.path.join(config.LOG_DIR, 'test_results.txt'), test_results)

    dices_val = np.concatenate(
        [np.expand_dims(np.asanyarray(val_files), -1).astype(np.str), dices_val.astype(np.str)], axis=1)
    dices_test = np.concatenate(
        [np.expand_dims(np.asanyarray(test_files), -1).astype(np.str), dices_test.astype(np.str)], axis=1)

    np.savetxt(os.path.join(config.LOG_DIR, 'val_dices.txt'), dices_val, '%s')
    np.savetxt(os.path.join(config.LOG_DIR, 'test_dices.txt'), dices_test, '%s')


def dice_coefficient(pred, gt):
    """
    Computes dice coefficients between two masks
    :param pred: predicted masks - [0 ,1]
    :param gt: ground truth  masks - [0 ,1]
    :return: dice coefficient
    """
    d = (2 * np.sum(pred * gt) + 1) / ((np.sum(pred) + np.sum(gt)) + 1)

    return d


def dice_coefficient_batch(pred, gt, eer_thresh=0.5):
    dice_all = []
    n = pred.shape[0]
    for i in range(n):
        seg = pred[i, :, :]
        seg = (seg >= eer_thresh).astype(np.uint8)
        gtd = gt[i, :, :]

        d = dice_coefficient(seg, gtd)
        dice_all.append(d)

    return dice_all


def compute_dice_metric(preds, labels, eval_class_indices=None):
    """
    Evaluates the segmentation by computing dice coefficient
    :param preds: NxHxWxC prediction masks  where pixel values are between [0,1] or
    :param labels: NxHxWxC ground truth masks where pixel values are between [0,1]
    :param eval_class_indices, indices of classes to be evaluated, if None, all indices will be evaluated
    """

    if (eval_class_indices is None):
        eval_class_indices = range(preds.shape[3])

    evals = [dice_coefficient_batch(preds[:, :, :, i], labels[:, :, :, i]) for i in eval_class_indices]
    evals = [np.expand_dims(np.asanyarray(e), -1) for e in evals]
    dices = np.concatenate(evals, axis=1)  # ( N,C) matrix
    dices_mean = np.mean(dices, axis=0)  # (C,)

    res_verbose = ''
    for c, ev in zip(eval_class_indices, dices_mean):
        res_verbose += 'Class ' + str(c) + ' DC=' + str(ev) + '\n'

    return dices_mean, dices, res_verbose


def batch_predict(img_dir, out_dir, config, segmodel=None, image_size=None, image_extension='.png'):
    """
    Generates segmentation results visualization for all  images in a given folder

    :param segmodel: segmentation model
    :param img_dir: directory where source images are locate
    :param out_dir:  path to save output segmentatiions, visualization will be saved on outdir/viz directory
    :param N_classes: number of classes that model predicts. see configuration
    :param image_size: image will be resized to image_size before prediction
    :return:
    """

    N_classes = config.NUM_CLASSES

    if (segmodel is None):
        segmodel = load_tfkeras_model(config.MODEL_SAVE_DIR, file_name_prefix=config.NAME, model=None,
                                      custom_objects={})

    if (not os.path.exists(out_dir)):
        os.mkdir(out_dir)
    test_data = DirectoryImagesTest(img_dir, image_extension)
    test_ds = SegmentationData(data=test_data.data, loadLabels=False, shuffle=False, isRGB=config.IS_RGB)

    resizer = [df.imgaug.Resize(image_size, interp=cv2.INTER_NEAREST)] if image_size else []
    test_ds = df.AugmentImageComponent(test_ds, augmentors=resizer)
    test_ds = df.MapDataComponent(test_ds, lambda x: x / 255.0, index=0)
    test_ds = df.MapDataComponent(test_ds, lambda x: np.expand_dims(x, -1), index=0)
    test_ds = df.BatchData(test_ds, batch_size=np.min([16, test_ds.size()]))

    batch_iter = test_ds.get_data()

    vizdir = os.path.join(out_dir, 'viz')
    if (not os.path.exists(vizdir)): os.mkdir(vizdir)

    image_name = lambda x: os.path.basename(x).split('.')[0]

    for batch in batch_iter:
        images, file_names = batch[0], batch[2]
        out = segmodel.predict(images, batch_size=4)
        images = images * 255

        out_labelmap = np.argmax(out, axis=3).astype(np.uint8)
        for l, f in zip(out_labelmap, file_names): cv2.imwrite(os.path.join(out_dir, image_name(f) + '.png'), l)

        viz, _ = visualize_labels_overlay_labelmap(np.argmax(out, axis=3), images, N_classes, stack_images=False)
        for vim, f in zip(viz, file_names): cv2.imwrite(os.path.join(vizdir, 'v' + f), vim)

    print('Done')


if (__name__ == '__main__'):
    labels = np.random.random((10, 50, 50, 9))
    preds = np.random.random((10, 50, 50, 9))
    dices_mean, dices, res_verbose = compute_dice_metric(preds, labels)
    print(res_verbose)
