

from keras.models import Model
from keras.layers import Dense
from transfer_models.ssr_net import SSR_net
from transfer_models.nasnet import mobile
# from transfer_models.xceptionnet import Xception
from keras.applications import ResNet50
import config
from utilities import get_datetime_str
from train import ModelTrainer
import warnings
warnings.filterwarnings("ignore", message="Your CPU supports")
from keras.utils import plot_model

from models import get_resnet50
model = get_resnet50(ignore_age_weights=True)


plot_model(model, to_file='model.png', show_shapes=True)
exit()
for lay in model.layers[-7:-1]:
    print '\n\n'
    print lay
    for d in dir(lay):
        # if d.startswith('output'):
        if d == 'output' or d == 'output_shape':
            print
            print d
            print getattr(lay, d)
