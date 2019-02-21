RESNET50_DEFAULT_IMG_WIDTH = 224
MARGIN = 0.20
TRAIN_BATCH_SIZE = 16
VALIDATION_SIZE = 100

# DATA_DIR contains training data, tensorboard logs, saved weights
DATA_DIR = 'data/'

ORIGINAL_IMGS_DIR = DATA_DIR+'images/'
ORIGINAL_IMGS_INFO_FILE = DATA_DIR+'data.csv'
CROPPED_IMGS_DIR = DATA_DIR+'cropped_imgs_0.20/'
CROPPED_IMGS_INFO_FILE = DATA_DIR+'normalized_data.csv'

LOG_DIR = DATA_DIR+'tb_logs/'

WEIGHTS_DIR = DATA_DIR+'saved_weights/'
RESNET50_AGE_WEIGHTS = WEIGHTS_DIR+'age_only_resnet50_weights.061-3.300-4.410.hdf5'
SSRNET_AGE_WEIGHTS = WEIGHTS_DIR+'ssrnet_3_3_3_64_1.0_1.0.h5'
