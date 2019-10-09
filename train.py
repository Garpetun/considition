import os

import matplotlib.pyplot as plt
from keras.optimizers import Adam
from segmentation_models.losses import bce_jaccard_loss, bce_dice_loss
from segmentation_models.metrics import iou_score

from datagenerator import DataGeneratorFolder, get_test_generator, get_train_generator
from callback import get_callbacks
from model import build_model


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

    # Xtest, ytest = test_generator.__getitem__(0)
    # plt.imshow(Xtest[0])
    # plt.show()
    # plt.imshow(ytest[0, :,:,0])
    # plt.show()

    train_generator  = get_train_generator()
    test_generator = get_test_generator()
    callbacks = get_callbacks()
    model = build_model()

    model.compile(optimizer=Adam(),
                  loss=bce_jaccard_loss, metrics=[iou_score])

    history = model.fit_generator(train_generator, shuffle=True,
                                  epochs=50, workers=4, use_multiprocessing=True,
                                  validation_data=test_generator,
                                  verbose=1, callbacks=callbacks)
    # plotting history
    plot_training_history(history)
