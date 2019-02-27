

from ModelTrainer import ModelTrainer
import warnings
warnings.filterwarnings("ignore", message="Your CPU supports")
from keras.utils import plot_model


from models import get_resnet50, get_inception_resnet_v2

# model = get_resnet50()
# trainer = ModelTrainer(model, 'RESNET50_even_samples')
# trainer.train_top_layer()
# trainer.train_all_layers()

# for cutoff in [200,300,400,None]:
for cutoff in [300]:
    # model = get_inception_resnet_v2()
    # trainer = ModelTrainer(model, 'INCEPTION_RESNET_V2_even_samples', cutoff)
    model = get_resnet50()
    model_name = 'RESNET50_%s_cutoff' % cutoff
    trainer = ModelTrainer(model, model_name, cutoff)
    trainer.train_top_layer()
    trainer.train_all_layers()

