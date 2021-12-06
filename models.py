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

    def loss(self, s_f):
        ## TODO: Verify that this matches what the paper is looking for 
        #return tf.math.log(s_f)
        binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        #return tf.reduce_mean(binary_cross_entropy(tf.ones_like(s_f), s_f))
        print(s_f.shape)
        print(tf.ones_like(s_f))
        return binary_cross_entropy(tf.ones_like(s_f), s_f)


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
            Conv2D(1, [4,4], activation='sigmoid')
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
        x = tf.reshape(x, shape=[])
        return x



    def loss(self, s_r, s_w, s_f):
        ## TODO: Finish filling out
        #return tf.math.log(s_r) + (tf.math.log(1 - s_w) + tf.math.log(1 - s_f))/2
        alpha = 0.5
        binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_output_noise = tf.ones_like(s_r)
        fake_output_real_text_noise_1 = tf.zeros_like(s_f)
        real_output_fake_text_noise = tf.zeros_like(s_w)

        #real_loss = tf.reduce_mean(binary_cross_entropy(real_output_noise, s_r))
        #fake_loss_ms_1 = tf.reduce_mean(binary_cross_entropy(fake_output_real_text_noise_1, s_f))
        #fake_loss_2 = tf.reduce_mean(binary_cross_entropy(real_output_fake_text_noise, s_w))
        real_loss = binary_cross_entropy(real_output_noise, s_r)
        fake_loss_ms_1 = binary_cross_entropy(fake_output_real_text_noise_1, s_f)
        fake_loss_2 = binary_cross_entropy(real_output_fake_text_noise, s_w)
        total_loss = real_loss + alpha * fake_loss_2 + (1-alpha) * fake_loss_ms_1 
        return total_loss