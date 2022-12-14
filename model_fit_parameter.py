from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def save_model():

    model_name = "model.h5"
    checkpoint = ModelCheckpoint(model_name,
                                 monitor="val_loss",
                                 mode="min",
                                 save_best_only=True,
                                 verbose=1)

    earlystopping = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=5, verbose=1, restore_best_weights=True)

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                                patience=3,
                                                verbose=1,
                                                factor=0.2,
                                                min_lr=0.00000001)

    return checkpoint, earlystopping, learning_rate_reduction
