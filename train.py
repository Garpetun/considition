import os

import matplotlib.pyplot as plt
from keras.optimizers import Adam
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

from datagenerator import get_test_generator, get_train_generator
from callback import get_callbacks
from model import build_model, load_model


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


def main():
    encoder = 'efficientnetb2'
    name = 'consid.' + encoder + '_1024_noaug'
    train_generator = get_train_generator(512)
    # for Xtest, ytest in train_generator:
    #     plt.subplot(121)
    #     plt.imshow(Xtest[0])
    #     plt.subplot(122)
    #     plt.imshow(ytest[0]*255)
    #     plt.show()
    test_generator = get_test_generator(1024)
    callbacks = get_callbacks(name)
    #model = build_model(encoder)
    model = load_model('consid.efficientnetb2_512')
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss=bce_jaccard_loss, metrics=[iou_score])

    history = model.fit_generator(train_generator, shuffle=True,
                                  epochs=100, workers=4, use_multiprocessing=True,
                                  validation_data=test_generator,
                                  verbose=1, callbacks=callbacks)
    # plotting history
    plot_training_history(history)


if __name__ == "__main__":
    main()