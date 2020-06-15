import numpy as np
from copy import deepcopy

import utils

import keras.backend
from keras.models import Model, Sequential
from keras.layers import Input, Add, Dense, Dropout, Flatten
from keras.layers import Activation, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers 
from keras.optimizers import Adam, Nadam
from keras.preprocessing.image import ImageDataGenerator

from keras.layers.normalization import BatchNormalization

import os
import tensorflow as tf

# Hide Tensorflow INFOS and WARNINGS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

class Particle:
    def __init__(self, min_layer, max_layer, max_pool_layers, input_width, input_height, input_channels, \
        conv_prob, pool_prob, fc_prob, max_conv_kernel, max_out_ch, max_fc_neurons, output_dim):
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels

        self.num_pool_layers = 0
        self.max_pool_layers = max_pool_layers

        self.feature_width = input_width
        self.feature_height = input_height

        self.depth = np.random.randint(min_layer, max_layer)
        self.conv_prob = conv_prob
        self.pool_prob = pool_prob
        self.fc_prob = fc_prob
        self.max_conv_kernel = max_conv_kernel
        self.max_out_ch = max_out_ch
        
        self.max_fc_neurons = max_fc_neurons
        self.output_dim = output_dim

        self.layers = []
        self.acc = None
        self.vel = [] # Initial velocity
        self.pBest = []

        # Build particle architecture
        self.initialization()
        
        # Update initial velocity
        for i in range(len(self.layers)):
            if self.layers[i]["type"] != "fc":
                self.vel.append({"type": "keep"})
            else:
                self.vel.append({"type": "keep_fc"})
        
        self.model = None
        self.pBest = deepcopy(self)

    
    def __str__(self):
        string = ""
        for z in range(len(self.layers)):
            string = string + self.layers[z]["type"] + " | "
        
        return string

    def initialization(self):
        out_channel = np.random.randint(3, self.max_out_ch)
        conv_kernel = np.random.randint(3, self.max_conv_kernel)
        
        # First layer is always a convolution layer
        self.layers.append({"type": "conv", "ou_c": out_channel, "kernel": conv_kernel})

        conv_prob = self.conv_prob
        pool_prob = conv_prob + self.pool_prob
        fc_prob = pool_prob

        for i in range(1, self.depth):
            if self.layers[-1]["type"] == "fc":
                layer_type = 1.1
            else:
                layer_type = np.random.rand()

            if layer_type < conv_prob:
                self.layers = utils.add_conv(self.layers, self.max_out_ch, self.max_conv_kernel)

            elif layer_type >= conv_prob and layer_type <= pool_prob:
                self.layers, self.num_pool_layers = utils.add_pool(self.layers, self.fc_prob, self.num_pool_layers, self.max_pool_layers, self.max_out_ch, self.max_conv_kernel, self.max_fc_neurons, self.output_dim)
            
            elif layer_type >= fc_prob:
                self.layers = utils.add_fc(self.layers, self.max_fc_neurons)
            
        self.layers[-1] = {"type": "fc", "ou_c": self.output_dim, "kernel": -1}
    

    def velocity(self, gBest, Cg):
        self.vel = utils.computeVelocity(gBest, self.pBest.layers, self.layers, Cg)

    def update(self):
        new_p = utils.updateParticle(self.layers, self.vel)
        new_p = self.validate(new_p)
        
        self.layers = new_p
        self.model = None

    def validate(self, list_layers):
        # Last layer should always be a fc with number of neurons equal to the number of outputs
        list_layers[-1] = {"type": "fc", "ou_c": self.output_dim, "kernel": -1}

        # Remove excess of Pooling layers
        self.num_pool_layers = 0
        for i in range(len(list_layers)):
            if list_layers[i]["type"] == "max_pool" or list_layers[i]["type"] == "avg_pool":
                self.num_pool_layers += 1
            
                if self.num_pool_layers >= self.max_pool_layers:
                    list_layers[i]["type"] = "remove"


        # Now, fix the inputs of each conv and pool layers
        updated_list_layers = []
        
        for i in range(0, len(list_layers)):
            if list_layers[i]["type"] != "remove":
                if list_layers[i]["type"] == "conv":
                    updated_list_layers.append({"type": "conv", "ou_c": list_layers[i]["ou_c"], "kernel": list_layers[i]["kernel"]})
                
                if list_layers[i]["type"] == "fc":
                    updated_list_layers.append(list_layers[i])

                if list_layers[i]["type"] == "max_pool":
                    updated_list_layers.append({"type": "max_pool", "ou_c": -1, "kernel": 2})

                if list_layers[i]["type"] == "avg_pool":
                    updated_list_layers.append({"type": "avg_pool", "ou_c": -1, "kernel": 2})

        return updated_list_layers

    ##### Model methods ####
    def model_compile(self, dropout_rate):
        list_layers = self.layers
        self.model = Sequential()

        for i in range(len(list_layers)):
            if list_layers[i]["type"] == "conv":
                n_out_filters = list_layers[i]["ou_c"]
                kernel_size = list_layers[i]["kernel"]

                if i == 0:
                    in_w = self.input_width
                    in_h = self.input_height
                    in_c = self.input_channels
                    self.model.add(Conv2D(n_out_filters, kernel_size, strides=(1,1), padding="same", data_format="channels_last", kernel_initializer='he_normal', bias_initializer='he_normal', activation=None, input_shape=(in_w, in_h, in_c)))
                    self.model.add(BatchNormalization())
                    self.model.add(Activation("relu"))
                else:
                    self.model.add(Dropout(dropout_rate))
                    self.model.add(Conv2D(n_out_filters, kernel_size, strides=(1,1), padding="same", kernel_initializer='he_normal', bias_initializer='he_normal', activation=None))
                    self.model.add(BatchNormalization())
                    self.model.add(Activation("relu"))

            if list_layers[i]["type"] == "max_pool":
                kernel_size = list_layers[i]["kernel"]

                self.model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

            if list_layers[i]["type"] == "avg_pool":
                kernel_size = list_layers[i]["kernel"]

                self.model.add(AveragePooling2D(pool_size=(3, 3), strides=2))
            
            if list_layers[i]["type"] == "fc":
                if list_layers[i-1]["type"] != "fc":
                    self.model.add(Flatten())

                self.model.add(Dropout(dropout_rate))

                if i == len(list_layers) - 1:
                    self.model.add(Dense(list_layers[i]["ou_c"], kernel_initializer='he_normal', bias_initializer='he_normal', activation=None))
                    self.model.add(BatchNormalization())
                    self.model.add(Activation("softmax"))
                else:
                    self.model.add(Dense(list_layers[i]["ou_c"], kernel_initializer='he_normal', bias_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01), activation=None))
                    self.model.add(BatchNormalization())
                    self.model.add(Activation("relu"))

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0)

        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
    

    def model_fit(self, x_train, y_train, batch_size, epochs):
        # TODO: add option to only use a sample size of the dataset

        hist = self.model.fit(x=x_train, y=y_train, validation_split=0.0, batch_size=batch_size, epochs=epochs)

        return hist

    def model_fit_complete(self, x_train, y_train, batch_size, epochs):
        hist = self.model.fit(x=x_train, y=y_train, validation_split=0.0, batch_size=batch_size, epochs=epochs)

        return hist
    
    def model_delete(self):
        # This is used to free up memory during PSO training
        del self.model
        keras.backend.clear_session()
        self.model = None