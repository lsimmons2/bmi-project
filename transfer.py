
from keras.models import Model
from keras.layers import Dense
from transfer_models.ssr_net import SSR_net
from transfer_models.nasnet import mobile
from keras.applications import ResNet50
import config
from utilities import get_datetime_str
from train import ModelTrainer
import warnings
warnings.filterwarnings("ignore", message="Your CPU supports")


img_size = 64
stage_num = [3,3,3]
lambda_local = 1
lambda_d = 1

# RESNET50
# model = ResNet50()

# SSR
# model = SSR_net(img_size,stage_num, lambda_local, lambda_d)()
# model.load_weights(config.SSRNET_AGE_WEIGHTS)
# model_name = 'SSRNET'
# print model.layers[-1].output
# trainer = ModelTrainer(model, model_name)
# trainer.train_top_layer()
# trainer.train_all_layers()

# NASNET
base_model = mobile(weights='imagenet')
last_hidden_layer = base_model.get_layer(index=-2)

base_model = Model(
    inputs=base_model.input,
    outputs=last_hidden_layer.output)

prediction = Dense(1, kernel_initializer='normal')(base_model.output)
model = Model(inputs=base_model.input, outputs=prediction)
model_name = 'NASNET'
print model.layers[-1].output
trainer = ModelTrainer(model, model_name)
trainer.train_top_layer()
trainer.train_all_layers()

# DENSENET
# import densenet

# # 'th' dim-ordering or 'tf' dim-ordering
# image_dim = (3, 32, 32) or image_dim = (32, 32, 3)

# model = densenet.DenseNet(classes=10, input_shape=image_dim, depth=40, growth_rate=12, bottleneck=True, reduction=0.5)
