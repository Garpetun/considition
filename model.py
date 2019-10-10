import keras.models
from segmentation_models.losses import bce_jaccard_loss, bce_dice_loss
from segmentation_models.metrics import iou_score
from segmentation_models import Unet


def build_model(encoder):
    return Unet(classes=3, backbone_name=encoder, encoder_weights='imagenet', encoder_freeze=False,
                decoder_filters=(256, 192, 128, 64, 32))


def load_model(model_name):
    return keras.models.load_model('./weights/'+model_name+'.h5',
                                   custom_objects={'binary_crossentropy_plus_jaccard_loss': bce_jaccard_loss,
                                                   'iou_score': iou_score})
