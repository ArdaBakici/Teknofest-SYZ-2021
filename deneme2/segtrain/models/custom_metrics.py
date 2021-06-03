import os
os.environ['KERAS_BACKEND'] = "plaidml.keras.backend"
from keras import backend as K
import  numpy as np

smooth = 1.


def softmax(x, axis):
    e = K.exp(x - K.max(x, axis=axis, keepdims=True))
    s = K.sum(e, axis=axis, keepdims=True)
    return e / s



def multiclass_dice_coef_metric(from_logits=True, class_index=None):
    def dice_coef(y_true, y_pred):
        if(from_logits):
           y_pred= softmax(y_pred, axis=3)

        if (class_index is not None):
            y_pred = y_pred[:,:, :, class_index]
            y_true = y_true[:,:,:,  class_index]

        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice_coef







def multiclass_balanced_cross_entropy_(y_true, y_pred, from_logits=False, pixelwise_loss_extra=None):
    loss_func = multiclass_balanced_cross_entropy(from_logits, pixelwise_loss_extra=pixelwise_loss_extra)
    return loss_func(y_true, y_pred)


def multiclass_balanced_cross_entropy(from_logits=False, pixelwise_loss_extra=None, use_entropy_penalty=False, P=None):
    """
    Returns a function that computes class balanced cross entropy for multi channel output
    :param from_logits:
    :param pixelwise_loss_extra: N,H,W,C extra loss pixel to include in
    :param use_entropy_penalty:
    :param P:  Parameter used during class balancing.
    :return:
    """
    def inner_func(y_true, y_pred):
        """

        :param y_true: ground truuth masks in channel last format
        :param y_pred: model predictios  in channel last format
        :return:
        """


        shape = K.shape(y_pred)
        batch_size = shape[0]
        num_classes = shape[3]

        if (from_logits):
            # using  tensorflow equation works just fine
            cross_ent = -1 * (K.maximum(y_pred, 0) - y_pred * y_true + K.log(1 + K.exp(-K.abs(y_pred))))  # BxHxWxC
            if (pixelwise_loss_extra is not None):
                cross_ent = cross_ent +pixelwise_loss_extra               
            
        else:
            y_pred_ = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
            cross_ent = (K.log(y_pred_) * y_true)  # BxHxWxC
        
       
            
        if (use_entropy_penalty):
            prob_dist= softmax(y_pred,axis=3)
            ent_penalty = -1 * (K.maximum(y_pred, 0) - y_pred * prob_dist + K.log(1 + K.exp(-K.abs(y_pred))))  # BxHxWxC
            #ent_penalty = -K.sum(ent_penalty, axis=[1, 2, 3], keepdims=False) #(B,)
            cross_ent   = cross_ent + 0.2* ent_penalty
        
        

        cross_ent = K.sum(cross_ent, axis=2, keepdims=False)  # Bx HxC
        cross_ent = K.sum(cross_ent, axis=1, keepdims=False)  # Bx C
        cross_ent = K.reshape(cross_ent, (batch_size, num_classes))

        y_true_ = K.sum(y_true, axis=2, keepdims=False)
        y_true_ = K.sum(y_true_, axis=1, keepdims=False)
        y_true_ = K.reshape(y_true_, (batch_size, num_classes)) + K.epsilon()

        if(P is not None):
            temp_numerator = np.where(K.less(y_true_, P), K.zeros_like(y_true_), K.ones_like(y_true_))
            temp_denom = np.where(K.less(y_true_, P), K.ones_like(y_true_), y_true_)
            cross_ent = temp_numerator * (cross_ent / temp_denom)  # (B, C)
        else:
            cross_ent = cross_ent / y_true_  # (B, C)

        cross_ent = - K.mean(cross_ent, axis=1)  # (B,)



        return cross_ent

    return inner_func

