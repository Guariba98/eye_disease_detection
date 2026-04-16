import keras
from keras import layers
from keras.applications import ResNet50, EfficientNetB0
from keras.models import Model

def create_odir_model(model_type='resnet50', num_classes=5, freeze_features=True):
    if model_type == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_type == 'efficientnetb0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    base_model.trainable = not freeze_features

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model