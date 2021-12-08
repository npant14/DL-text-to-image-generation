import tensorflow as tf
from models import Generator, Discriminator
from preprocessing import get_data
import random
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def train(gen_model, dis_model, imgs, captions):

    total_gen_loss = 0
    total_dis_loss = 0

    iter = 0
    batch_size = 100

    for i in range(0, batch_size * (imgs.shape[0] // batch_size), batch_size):
        iter += 1
        print("number " + str(iter) + " with losses: " + str(total_dis_loss) + " (dis loss), " + str(total_gen_loss) + " (gen loss)")
        z = tf.random.normal([batch_size, 128])
        caps = captions[i: min(i+batch_size, imgs.shape[0])]
        
        with tf.GradientTape() as tape:
            ## update gradients for generator model 
            fimg = gen_model(z, caps)
            rcap = captions[np.random.randint(captions.shape[0], size=(batch_size)),:]
            s_f = dis_model(fimg, caps)
            gen_loss = gen_model.loss(s_f)
            gen_gradients = tape.gradient(tf.reduce_mean(gen_loss), gen_model.trainable_variables)
            gen_model.optimizer.apply_gradients(zip(gen_gradients, gen_model.trainable_variables))
            total_gen_loss += tf.reduce_sum(gen_loss)

        with tf.GradientTape() as tape:
            ## update gradients for discriminator model 
            fimg = gen_model(z, caps)
            rcap = captions[np.random.randint(captions.shape[0], size=(batch_size)),:]
            
            ## real images with real captions
            s_r = dis_model(imgs[i: min(i+batch_size, imgs.shape[0])], caps)
            ## real images with random captions
            s_w = dis_model(imgs[i: min(i+batch_size, imgs.shape[0])], rcap)
            ## fake images with real captions
            s_f = dis_model(fimg, caps)
            dis_loss = dis_model.loss(s_r, s_w, s_f)
            dis_gradients = tape.gradient(tf.reduce_mean(dis_loss), dis_model.trainable_variables)
            dis_model.optimizer.apply_gradients(zip(dis_gradients, dis_model.trainable_variables))
            total_dis_loss += tf.reduce_sum(dis_loss)

    return total_gen_loss, total_dis_loss


def test(gen_model, dis_model, imgs, captions):
    total_gen_loss = 0
    total_dis_loss = 0
    for img, caption in zip(imgs, captions):
            fimg = gen_model(img, caption)
            rcap = captions[random.randint(0, len(captions))]
            s_r = dis_model(img, caption)
            s_w = dis_model(img, rcap)
            s_f = dis_model(fimg, caption)
            binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            gen_loss = tf.reduce_mean(binary_cross_entropy(tf.ones_like(s_f), s_f))

            alpha = 0.5
            real_output_noise = tf.ones_like(s_r)
            fake_output_real_text_noise_1 = tf.zeros_like(s_f)
            real_output_fake_text_noise = tf.zeros_like(s_w)

            real_loss = tf.reduce_mean(binary_cross_entropy(real_output_noise, s_r))
            fake_loss_ms_1 = tf.reduce_mean(binary_cross_entropy(fake_output_real_text_noise_1, s_f))
            fake_loss_2 = tf.reduce_mean(binary_cross_entropy(real_output_fake_text_noise, s_w))

            dis_loss = real_loss + alpha * fake_loss_2 + (1-alpha) * fake_loss_ms_1 
            
            total_gen_loss += gen_loss
            total_dis_loss += dis_loss
    return total_gen_loss, total_dis_loss

def save_model_weights(model, path):
    """
    save weights of the model to specified path so they can
    be loaded in at another point in time
    """
    model.save_weights(path)


def load_model_weights(model, path):
    """
    for loading in pretrained weights for the model
    """
    model.load_weights(path)

def visualize_generation_results(gen_model, dis_model, captions):
    """
    param gen_model : the trained generator model 
    param dis_model : the trained discriminator model
    param captions  : a list of captions

    Takes in the trained models + captions and generates the associated
    visuals
    """
    n = len(captions)
        
    fig = plt.figure(figsize=(1, n))

    gspec = gridspec.GridSpec(1, n)
    gspec.update(wspace=0.5, hspace=0.5)

    for idx, cap in enumerate(captions):
        z = tf.random.normal([1, 128])
        print('vis caption shape', np.expand_dims(cap, axis=0).shape)
        generated_img = gen_model(z, tf.expand_dims(cap, axis=0)).numpy()
        score = dis_model(generated_img, tf.expand_dims(cap, axis=0)).numpy()


        ax = plt.subplot(gspec[idx])
        ## print out discriminator score
        print(score)

        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        ax.set_xlabel(score)
        plt.imshow(tf.reshape(generated_img, (64, 64, 3)))

    plt.show()

    ## save to pdf (in case plt interactive window doesn't work)
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", "visualize_results_smooth_200_lr.pdf")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ## get data
    (train_images, train_captions) = get_data()
    print(train_images.shape)
    print(train_captions.shape)

    ## initialize models
    generator = Generator()
    discriminator = Discriminator()
    
    ## train models
    for i in range(200):
        print("starting epoch ", i)
        train(generator, discriminator, train_images, train_captions)


    ## if LOADING weights, train for 1 epoch with above
    ## then uncomment the two lines below

    # load_model_weights(generator, 'generator_weights_smooth.h5')
    # load_model_weights(discriminator, 'discriminator_weights_smooth.h5')
    
    save_model_weights(generator, 'generator_weights_smooth_200_lr.h5')
    save_model_weights(discriminator, 'discriminator_weights_smooth_200_lr.h5')
    visualize_generation_results(generator, discriminator, train_captions[:,10])

if __name__ == "__main__":

    main()