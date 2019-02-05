
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


def get_model(ignore_age_weights=False):

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
