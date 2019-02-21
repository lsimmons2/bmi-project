
from transfer_models.ssr_net import SSR_net
from tensorflow.python.keras.applications import ResNet50
import config
from utilities import get_datetime_str
from train import ModelTrainer

img_size = 64
stage_num = [3,3,3]
lambda_local = 1
lambda_d = 1

# model = ResNet50()

model = SSR_net(img_size,stage_num, lambda_local, lambda_d)()
model.load_weights(config.SSRNET_AGE_WEIGHTS)
model_name = 'SSRNET'
trainer = ModelTrainer(model, model_name)
trainer.train_top_layer()
trainer.train_all_layers()
