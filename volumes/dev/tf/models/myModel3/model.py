#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    20-Mar-2023 18:17:38

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model():
    featureInputLayer = keras.Input(shape=(186,))
    fullyConnectedLayer1 = layers.Dense(28, name="fullyConnectedLayer1_")(featureInputLayer)
    batchNormalizationLayer = layers.BatchNormalization(epsilon=0.000010, name="batchNormalizationLayer_")(fullyConnectedLayer1)
    sigmoidLayer = layers.Activation('sigmoid')(batchNormalizationLayer)
    dropoutLayer = layers.Dropout(0.150000)(sigmoidLayer)
    fullyConnectedLayer3 = layers.Dense(28, name="fullyConnectedLayer3_")(dropoutLayer)
    sigmoidLayer2 = layers.Activation('sigmoid')(fullyConnectedLayer3)
    fullyConnectedLayer4 = layers.Dense(14, name="fullyConnectedLayer4_")(sigmoidLayer2)
    softmaxLayer = layers.Softmax()(fullyConnectedLayer4)
    classificationLayer = softmaxLayer

    model = keras.Model(inputs=[featureInputLayer], outputs=[classificationLayer])
    return model
