from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard

def get_callbacks():

    # reduces learning rate on plateau
    lr_reducer = ReduceLROnPlateau(factor=0.1,
                                cooldown= 10,
                                patience=10,verbose =1,
                                min_lr=0.1e-5)
    mode_autosave = ModelCheckpoint("./weights/road_crop.efficientnetb0imgsize.h5",monitor='val_iou_score', 
                                    mode = 'max', save_best_only=True, verbose=1, period =10)

    # stop learining as metric on validatopn stop increasing
    early_stopping = EarlyStopping(patience=10, verbose=1, mode = 'auto') 

    # tensorboard for monitoring logs
    tensorboard = TensorBoard(log_dir='./logs/tenboard', histogram_freq=0,
                            write_graph=True, write_images=False)

    callbacks = [mode_autosave, lr_reducer, tensorboard, early_stopping]

    return callbacks