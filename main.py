import os

import matplotlib.pyplot as plt

from datagenerator import DataGeneratorFolder
from augmentation import aug_with_crop
from callback import get_callbacks

DATASET_DIR = os.path.join('.', 'data', 'consid')
IMAGE_FOLDER = 'Images'
MASKS_FOLDER = os.path.join('Masks', 'all')


def plot_training_history(history):
    """
    Plots model training history
    """
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15, 5))
    ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
    ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax_loss.legend()
    ax_acc.plot(history.epoch, history.history["iou_score"], label="Train iou")
    ax_acc.plot(history.epoch, history.history["val_iou_score"], label="Validation iou")
    ax_acc.legend()


if __name__ == "__main__":
    # # Benchmark dataset
    # test_generator = DataGeneratorFolder(root_dir = 'data/consid/',
    #                                      image_folder = 'Images/',
    #                                      mask_folder = 'Masks/',
    #                                      batch_size = 1,
    #                                      nb_y_features = 1, augmentation = aug_with_crop)

    # Consid dataset
    test_generator = DataGeneratorFolder(root_dir = DATASET_DIR,
                                         image_folder = IMAGE_FOLDER,
                                         mask_folder = MASKS_FOLDER,
                                         batch_size = 10,
                                         nb_y_features = 3,
                                         augmentation = aug_with_crop)
    # Xtest, ytest = test_generator.__getitem__(0)
    # plt.imshow(Xtest[0])
    # plt.show()
    # plt.imshow(ytest[0, :,:,0])
    # plt.show()

    train_generator = DataGeneratorFolder(root_dir = DATASET_DIR,
                                        image_folder = IMAGE_FOLDER,
                                        mask_folder = MASKS_FOLDER,
                                        augmentation = aug_with_crop,
                                        batch_size=4,
                                        image_size=512,
                                        nb_y_features = 3)

    callbacks = get_callbacks()

    from segmentation_models import Unet
    from keras.optimizers import Adam
    from segmentation_models.losses import bce_jaccard_loss, bce_dice_loss
    from segmentation_models.metrics import iou_score

    model = Unet(classes=3, backbone_name='efficientnetb0', encoder_weights='imagenet', encoder_freeze=False)
    model.compile(optimizer=Adam(),
                  loss=bce_jaccard_loss, metrics=[iou_score])

    history = model.fit_generator(train_generator, shuffle=True,
                                  epochs=50, workers=4, use_multiprocessing=True,
                                  validation_data=test_generator,
                                  verbose=1, callbacks=callbacks)
    # plotting history
    plot_training_history(history)
