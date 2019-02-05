
import cv2
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from train_generator import train_generator, plot_imgs_from_generator
from mae_callback import MAECallback
import config



batches_per_epoch=train_generator.n //train_generator.batch_size


def train_top_layer(model):

    print 'Training top layer...'

    for l in model.layers[:-1]:
        l.trainable = False

    model.compile(
        loss='mean_squared_error',
        optimizer='adam'
    )

    mae_callback = MAECallback()

    early_stopping_callback = EarlyStopping(
        monitor='val_mae',
        mode='min',
        verbose=1,
        patience=1)

    model_checkpoint_callback = ModelCheckpoint(
        'saved_models/top_layer_trained_weights.{epoch:02d}-{val_mae:.2f}.h5',
        monitor='val_mae',
        mode='min',
        verbose=1,
        save_best_only=True
    )

    tensorboard_callback = TensorBoard(
        log_dir=config.TOP_LAYER_LOG_DIR,
        batch_size=train_generator.batch_size
    )

    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=batches_per_epoch,
        epochs=20,
        callbacks=[
            mae_callback,
            early_stopping_callback,
            model_checkpoint_callback,
            tensorboard_callback
        ]
    )


def train_all_layers(model):

    print 'Training all layers...'

    for l in model.layers:
        l.trainable = True

    mae_callback = MAECallback()

    early_stopping_callback = EarlyStopping(
        monitor='val_mae',
        mode='min',
        verbose=1,
        patience=10)

    model_checkpoint_callback = ModelCheckpoint(
        'saved_models/all_layers_trained_weights.{epoch:02d}-{val_mae:.2f}.h5',
        monitor='val_mae',
        mode='min',
        verbose=1,
        save_best_only=True)

    tensorboard_callback = TensorBoard(
        log_dir=config.ALL_LAYERS_LOG_DIR,
        batch_size=train_generator.batch_size
    )

    model.compile(
        loss='mean_squared_error',
        optimizer='adam'
    )

    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=batches_per_epoch,
        epochs=100,
        callbacks=[
            mae_callback,
            early_stopping_callback,
            model_checkpoint_callback,
            tensorboard_callback
        ]
    )


def test_model(model):

    with open(config.CROPPED_IMGS_INFO_FILE, 'r') as f:
        test_images_info = f.read().splitlines()[-config.VALIDATION_SIZE:]

    test_X = []
    test_y = []
    for info in test_images_info:
        bmi = float(info.split(',')[1])
        test_y.append(bmi)
        file_name = info.split(',')[4]
        file_path = '%s/%s' % (config.CROPPED_IMGS_DIR, file_name)
        img = cv2.imread(file_path)
        img = cv2.resize(img, (config.RESNET50_DEFAULT_IMG_WIDTH, config.RESNET50_DEFAULT_IMG_WIDTH))
        test_X.append(img)

    test_X = np.array(test_X)
    test_y = np.array(test_y)

    mae = get_mae(test_y, model.predict(test_X))
    print '\nMAE:', mae
