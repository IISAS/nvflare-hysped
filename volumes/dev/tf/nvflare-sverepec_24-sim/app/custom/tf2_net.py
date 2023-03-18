# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf


class Net(tf.keras.Model):
        
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.num_classes = num_classes
        self.fullyConnectedLayer1 = tf.keras.layers.Dense(2 * num_classes, name='fullyConnectedLayer1')
        self.batchNormalizationLayer = tf.keras.layers.BatchNormalization(name='batchNormalizationLayer')
        self.sigmoidLayer = tf.keras.layers.Activation(tf.keras.activations.sigmoid, name='sigmoidLayer')
        self.dropoutLayer = tf.keras.layers.Dropout(0.15, name='dropoutLayer')
        self.fullyConnectedLayer3 = tf.keras.layers.Dense(2 * num_classes, name='fullyConnectedLayer3')
        self.sigmoidLayer2 = tf.keras.layers.Activation(tf.keras.activations.sigmoid, name='sigmoidLayer2')
        self.fullyConnectedLayer4 = tf.keras.layers.Dense(num_classes, name='fullyConnectedLayer4')
        self.softmaxLayer = tf.keras.layers.Softmax(name='softmaxLayer')

    def call(self, x):
        x = self.fullyConnectedLayer1(x)
        x = self.batchNormalizationLayer(x)
        x = self.sigmoidLayer(x)
        x = self.dropoutLayer(x)
        x = self.fullyConnectedLayer3(x)
        x = self.sigmoidLayer2(x)
        x = self.fullyConnectedLayer4(x)
        x = self.softmaxLayer(x)
        return x
    
    def get_config(self):
        return {
            'num_classes': self.num_classes
        }
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def compileModel(input_shape, num_classes):
        input_layer = tf.keras.layers.Input(shape=input_shape)
        x = Net(num_classes)(input_layer)

        model = tf.keras.models.Model(
            inputs=input_layer,
            outputs=x
        )
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy']
        )
        return model
