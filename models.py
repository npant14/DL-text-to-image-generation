import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Reshape, LeakyReLU, BatchNormalization
from tensorflow.math import log

## generator
class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        
        self.text_embedding = Sequential([
            Dense(128, activation=None), 
            LeakyReLU(alpha=0.05)])
        ## explore leaky relu activations

        self.deconv = Sequential([
            Dense(8*8*256),
            Reshape((8, 8, 256)),
            Conv2DTranspose(128, [5,5], strides=(1, 1), padding='same', activation='relu'),
            BatchNormalization(),
            Conv2DTranspose(64, [5,5], strides=(2, 2), padding='same', activation='relu'),
            BatchNormalization(),
            Conv2DTranspose(32, [5,5], strides=(2, 2), padding='same', activation='relu'),
            BatchNormalization(),
            Conv2DTranspose(3, [5,5], strides=(2, 2), padding='same', activation='tanh'),
            ])

    def call(self, latent_rep, text):
        """
        param latent_rep: latent space representation to be turned into image
        param text      : text embedding to concat with image

        returns: fake generated image
        """
        embedded_text = self.text_embedding(text)
        x = tf.concat([latent_rep, embedded_text], axis=-1)
        fimg = self.deconv(x)
        return fimg

    def loss(fake_output):
        ## TODO: Verify that this matches what the paper is looking for 
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return bce(tf.ones_like(fake_output), fake_output)


## discriminator
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

        self.text_embedding = Sequential([
            Dense(128, activation=None), 
            LeakyReLU(alpha=0.05)])

        self.conv = Sequential([
            Conv2D(16, [5,5], strides = (2,2), activation='relu'),
            BatchNormalization(),
            Conv2D(32, [3,3], strides = (2,2), activation='relu'),
            BatchNormalization(),
            Conv2D(64, [3,3], strides = (2,2), activation='relu'),
            BatchNormalization(),
            Conv2D(128, [3,3], strides = (1,1), activation=None)
        ])

        self.fc = Sequential([
            Conv2D(128, [1,1], activation='relu'),
            ## another rectification here
            BatchNormalization(), 
            Conv2D(1, [4,4])
        ])

    def call(self, img, text):
        """
        param latent_rep: latent space representation to be turned into image
        param text      : text embedding to concat with image

        returns: probability that the image is from the training set
        """
        four_by_four =  self.conv(img)

        embedded_text = self.text_embedding(text)
        [fdim, sdim] = embedded_text.shape
        embedded_text = tf.reshape(embedded_text, [-1, 1, fdim, sdim])
        embedded_text = tf.tile(embedded_text, [1,4,4,1])
        x = tf.concat([four_by_four, embedded_text], axis=-1)
        x = self.fc(x)
        return x



    def loss(out):
        ## TODO: Finish filling out
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        pass