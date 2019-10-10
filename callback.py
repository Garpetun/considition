from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard


def get_callbacks(model_name):
    # reduces learning rate on plateau
    lr_reducer = ReduceLROnPlateau(factor=0.2,
                                   cooldown= 10,
                                   patience=5,
                                   verbose =1,
                                   min_lr=0.1e-6)
    mode_autosave = ModelCheckpoint("./weights/"+model_name+".h5", monitor='val_iou_score',
                                    mode = 'max', save_best_only=True, verbose=1, period =5)

    # stop learining as metric on validatopn stop increasing
    early_stopping = EarlyStopping(patience=20, verbose=1, mode = 'auto') 

    # tensorboard for monitoring logs
    tensorboard = TensorBoard(log_dir='./logs/tenboard', histogram_freq=0,
                              write_graph=True, write_images=False)

    callbacks = [mode_autosave, lr_reducer, tensorboard, early_stopping]

    return callbacks