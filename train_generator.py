
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import Augmentor
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import config


def plot_imgs_from_generator(generator, number_imgs_to_show=9):
    print ('Plotting images...')
    n_rows_cols = int(math.ceil(math.sqrt(number_imgs_to_show)))
    plot_index = 1
    x_batch, _ = next(generator)
    while plot_index <= number_imgs_to_show:
        plt.subplot(n_rows_cols, n_rows_cols, plot_index)
        plt.imshow(x_batch[plot_index-1])
        plot_index += 1
    plt.show()


def augment_image(np_img):
    p = Augmentor.Pipeline()
    p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
    p.flip_left_right(probability=0.5)
    # p.zoom_random(probability=0.5, percentage_area=0.95)
    p.random_distortion(probability=0.25, grid_width=2, grid_height=2, magnitude=8)
    p.random_color(probability=1, min_factor=0.8, max_factor=1.2)
    p.random_contrast(probability=.5, min_factor=0.8, max_factor=1.2)
    p.random_brightness(probability=1, min_factor=0.5, max_factor=1.5)

    image = [Image.fromarray(np_img.astype('uint8'))]
    for operation in p.operations:
        r = round(random.uniform(0, 1), 1)
        if r <= operation.probability:
            image = operation.perform_operation(image)
    image = [np.array(i).astype('float64') for i in image]
    return image[0]

image_processor = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=augment_image
)

# subtract validation size from training data
with open(config.CROPPED_IMGS_INFO_FILE) as f:
    for i, _ in enumerate(f):
        pass
    training_n = i - config.VALIDATION_SIZE

train_df=pd.read_csv(config.CROPPED_IMGS_INFO_FILE, nrows=training_n)

train_generator=image_processor.flow_from_dataframe(
    dataframe=train_df,
    directory=config.CROPPED_IMGS_DIR,
    x_col='name',
    y_col='bmi',
    class_mode='other',
    color_mode='rgb',
    target_size=(config.RESNET50_DEFAULT_IMG_WIDTH,config.RESNET50_DEFAULT_IMG_WIDTH),
    batch_size=config.TRAIN_BATCH_SIZE)
