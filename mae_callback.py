
from keras.callbacks import Callback
import numpy as np
import config
import cv2


def get_mae(actual, predicted):
    n_samples = predicted.shape[0]
    diff_sum = 0.00
    for i in range(n_samples):
        p = predicted[i][0]
        a = actual[i]
        d = abs(p - a)
        diff_sum += d
    return diff_sum / n_samples


class MAECallback(Callback):


    def on_train_begin(self, logs={}):
        self._data = []


    def on_epoch_end(self, batch, logs={}):
        with open(config.CROPPED_IMGS_INFO_FILE, 'r') as f:
            test_images_info = f.read().splitlines()[-config.VALIDATION_SIZE:]
        test_x = []
        test_y = []
        for info in test_images_info:
            bmi = float(info.split(',')[1])
            test_y.append(bmi)
            file_name = info.split(',')[4]
            file_path = '%s/%s' % (config.CROPPED_IMGS_DIR, file_name)
            print
            print
            print
            print file_path
            img = cv2.imread(file_path)
            img = cv2.resize(img, (config.RESNET50_DEFAULT_IMG_WIDTH,config.RESNET50_DEFAULT_IMG_WIDTH))
            test_x.append(img/255.00)
        X_val = np.array(test_x)
        y_val = np.array(test_y)
        y_predict = np.asarray(self.model.predict(X_val))
        val_mae = get_mae(y_val, y_predict)
        logs['val_mae'] = val_mae
        self._data.append({
            'val_mae': val_mae,
        })
        return


    def get_data(self):
        return self._data
