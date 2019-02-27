
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

class Quartile(object):

    def __init__(self, q_range):
        self.q_range = q_range

class MAECallback(Callback):

    def __init__(self, img_size):
        super(MAECallback, self).__init__()
        self.img_size = img_size

    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, batch, logs={}):
        with open('validation_samples_300_cutoff.csv', 'r') as f:
            test_images_info = f.read().splitlines()[1:]
        quartiles = [((0,20),[]),
                     ((20,30),[]),
                     ((30,40),[]),
                     ((40,100),[]),
                     ((0,100),[])]
        for info in test_images_info:
            bmi = float(info.split(',')[1])
            file_name = info.split(',')[4]
            file_path = '%s/%s' % (config.CROPPED_IMGS_DIR, file_name)
            img = cv2.imread(file_path)
            img = cv2.resize(img, self.img_size)
            img = img/255.0
            x = img
            y = bmi
            y_hat = self.model.predict(np.array([x]))[0]
            for quartile in quartiles:
                quartile_range = quartile[0]
                if bmi >= quartile_range[0] and bmi < quartile_range[1]:
                    quartile[1].append((y,y_hat))

        for i, q in enumerate(quartiles):
            q_y = np.array([s[0] for s in q[1]])
            q_y_hat = np.array([s[1] for s in q[1]])
            q_mae = get_mae(q_y, q_y_hat)
            log_q_name = 'val_mae_%d-%d' % (q[0][0],q[0][1])
            print log_q_name, len(q_y), len(q_y_hat)
            logs[log_q_name] = q_mae
        val_mae = q_mae # last quartile is everything

        cs2_sum = 0
        cs4_sum = 0
        all_predictions = quartiles[-1][1]
        for y,y_hat in all_predictions:
            diff = abs(y_hat - y)
            if diff < 4:
                cs4_sum += 1
            if diff < 2:
                cs2_sum += 1
        cs2 = (cs2_sum / float(len(all_predictions)))*100
        cs4 = (cs4_sum / float(len(all_predictions)))*100

        logs['cs2'] = cs2
        logs['cs4'] = cs4
        self._data.append({
            'val_mae': val_mae,
        })
        return

    def get_data(self):
        return self._data
