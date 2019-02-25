
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.layers import Dense
import config

def get_age_model():

    age_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(config.RESNET50_DEFAULT_IMG_WIDTH, config.RESNET50_DEFAULT_IMG_WIDTH, 3),
        pooling='avg'
    )

    prediction = Dense(units=101,
                       kernel_initializer='he_normal',
                       use_bias=False,
                       activation='softmax',
                       name='pred_age')(age_model.output)

    age_model = Model(inputs=age_model.input, outputs=prediction)
    return age_model


def get_resnet50(ignore_age_weights=False):

    base_model = get_age_model()
    if not ignore_age_weights:
        base_model.load_weights(config.AGE_TRAINED_WEIGHTS_FILE)
        print 'Loaded weights from age classifier'
    last_hidden_layer = base_model.get_layer(index=-2)

    base_model = Model(
        inputs=base_model.input,
        outputs=last_hidden_layer.output)
    prediction = Dense(1, kernel_initializer='normal')(base_model.output)

    model = Model(inputs=base_model.input, outputs=prediction)
    return model


img_size = 64
stage_num = [3,3,3]
lambda_local = 1
lambda_d = 1


# SSR
# model = SSR_net(img_size,stage_num, lambda_local, lambda_d)()
# model.load_weights(config.SSRNET_AGE_WEIGHTS)
# model_name = 'SSRNET'
# print model.layers[-1].output
# trainer = ModelTrainer(model, model_name)
# trainer.train_top_layer()
# trainer.train_all_layers()

# NASNET
# base_model = mobile(weights='imagenet')
# last_hidden_layer = base_model.get_layer(index=-2)
# base_model = Model(
    # inputs=base_model.input,
    # outputs=last_hidden_layer.output)
# prediction = Dense(1, kernel_initializer='normal')(base_model.output)
# model = Model(inputs=base_model.input, outputs=prediction) # model_name = 'NASNET'
# trainer = ModelTrainer(model, model_name)
# trainer.train_top_layer()
# trainer.train_all_layers()


# INCEPTION_RESNET_V2
# from keras.applications.inception_resnet_v2 import InceptionResNetV2
# base_model = InceptionResNetV2()
# last_hidden_layer = base_model.get_layer(index=-2)
# base_model = Model(
    # inputs=base_model.input,
    # outputs=last_hidden_layer.output)
# prediction = Dense(1, kernel_initializer='normal')(base_model.output)
# model = Model(inputs=base_model.input, outputs=prediction)
# model_name = 'INCEPTION_RESNET_V2'
# trainer = ModelTrainer(model, model_name)
# trainer.train_top_layer()
# trainer.train_all_layers()

# base_model = Xception
# print base_model.layers[-1]

# DENSENET
# import densenet

# # 'th' dim-ordering or 'tf' dim-ordering
# image_dim = (3, 32, 32) or image_dim = (32, 32, 3)

# model = densenet.DenseNet(classes=10, input_shape=image_dim, depth=40, growth_rate=12, bottleneck=True, reduction=0.5)
