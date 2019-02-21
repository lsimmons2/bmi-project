
import cv2
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from train_generator import plot_imgs_from_generator, image_processor
from mae_callback import MAECallback
import config
import pandas as pd
from utilities import get_datetime_str



class ModelTrainer(object):

    LOSS = 'mean_absolute_error'
    OPTIMIZER = 'adam'

    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        # does not want number of samples or number of channels
        self.model_input_size = model.input_shape[1:-1]
        self.file_name_base = '%s_%s' % (self.model_name, get_datetime_str())

    def _get_train_generator(self):
        train_df=pd.read_csv(config.CROPPED_IMGS_INFO_FILE, nrows=100)
        return image_processor.flow_from_dataframe(
            dataframe=train_df,
            directory=config.CROPPED_IMGS_DIR,
            x_col='name',
            y_col='bmi',
            class_mode='other',
            color_mode='rgb',
            target_size=self.model_input_size,
            batch_size=config.TRAIN_BATCH_SIZE)

    def train_top_layer(self):

        print 'Training top layer...'

        # weight and log files
        weights_path = '%s/%s_top_layer.h5' % (config.WEIGHTS_DIR,self.file_name_base)
        tb_log_dir = '%s/%s_top_layer' % (config.LOG_DIR, self.file_name_base)

        # freeze layers
        for l in self.model.layers[:-1]:
            l.trainable = False

        # recompile (necessary after freezing layers)
        self.model.compile(
            loss=self.LOSS,
            optimizer=self.OPTIMIZER)

        # instantiate callbacks
        mae_callback = MAECallback(self.model_input_size)

        early_stopping_callback = EarlyStopping(
            monitor='val_mae',
            mode='min',
            verbose=1,
            patience=1)

        model_checkpoint_callback = ModelCheckpoint(
            weights_path,
            monitor='val_mae',
            mode='min',
            verbose=1,
            save_best_only=True,
            save_weights_only=True)

        tensorboard_callback = TensorBoard(
            log_dir=tb_log_dir,
            batch_size=config.TRAIN_BATCH_SIZE)

        # create and fit to training data generator
        train_generator = self._get_train_generator()

        batches_per_epoch=train_generator.n //train_generator.batch_size
        self.model.fit_generator(
            generator=train_generator,
            steps_per_epoch=batches_per_epoch,
            epochs=10,
            callbacks=[
                mae_callback,
                early_stopping_callback,
                model_checkpoint_callback,
                tensorboard_callback])


    def train_all_layers(self):

        # weight and log files
        weights_path = '%s/%s_all_layers.h5' % (config.WEIGHTS_DIR,self.file_name_base)
        tb_log_dir = '%s/%s_all_layers' % (config.LOG_DIR, self.file_name_base)

        print 'Training all layers...'

        # unfreeze all layers
        for l in self.model.layers:
            l.trainable = True

        # recompile model after unfreezing layers
        self.model.compile(
            loss=self.LOSS,
            optimizer=self.OPTIMIZER)

        # instantiate callbacks
        mae_callback = MAECallback(self.model_input_size)

        early_stopping_callback = EarlyStopping(
            monitor='val_mae',
            mode='min',
            verbose=1,
            patience=10)

        model_checkpoint_callback = ModelCheckpoint(
            weights_path,
            monitor='val_mae',
            mode='min',
            verbose=1,
            save_weights_only=True,
            save_best_only=True)

        tensorboard_callback = TensorBoard(
            log_dir=tb_log_dir,
            batch_size=config.TRAIN_BATCH_SIZE)


        # create and fit to training data generator
        train_generator = self._get_train_generator()

        batches_per_epoch=train_generator.n //train_generator.batch_size
        self.model.fit_generator(
            generator=train_generator,
            steps_per_epoch=batches_per_epoch,
            epochs=10,
            callbacks=[
                mae_callback,
                early_stopping_callback,
                model_checkpoint_callback,
                tensorboard_callback])


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
