import tensorflow as tf
from tensorflow.python.ops.gen_batch_ops import batch
from models import Generator, Discriminator
from preprocessing import get_data
import random
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def train(gen_model, dis_model, imgs, captions):
    ## TODO: Write the training loop for 1 epoch of the model

    total_gen_loss = 0
    total_dis_loss = 0

    ## not batched?
    iter = 0
    batch_size = 100
    #for img, caption in zip(imgs, captions):
    for i in range(0, imgs.shape[0], batch_size):
        iter += 1
        print("number " + str(iter) + " with losses: " + str(total_dis_loss) + " (dis loss), " + str(total_gen_loss) + " (gen loss)")
        z = tf.random.normal([batch_size, 128])
        caps = captions[i: min(i+batch_size, imgs.shape[0])]
        
        with tf.GradientTape() as tape:
            print('caption batch shape', caps.shape)
            fimg = gen_model(z, caps)
            rcap = captions[np.random.randint(captions.shape[0], size=(batch_size)),:]
            #rcap = np.random.choice(captions, size=batch_size)
            s_r = dis_model(imgs[i: min(i+batch_size, imgs.shape[0])], caps)
            s_w = dis_model(imgs[i: min(i+batch_size, imgs.shape[0])], rcap)
            s_f = dis_model(fimg, caps)
            gen_loss = gen_model.loss(s_f)
            gen_gradients = tape.gradient(tf.reduce_mean(gen_loss), gen_model.trainable_variables)
            gen_model.optimizer.apply_gradients(zip(gen_gradients, gen_model.trainable_variables))
            total_gen_loss += tf.reduce_sum(gen_loss)
        with tf.GradientTape() as tape:
            fimg = gen_model(z, caps)
            rcap = captions[np.random.randint(captions.shape[0], size=(batch_size)),:]
            s_r = dis_model(imgs[i: min(i+batch_size, imgs.shape[0])], caps)
            s_w = dis_model(imgs[i: min(i+batch_size, imgs.shape[0])], rcap)
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
            #gen_loss = tf.math.log(s_f)
            binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            gen_loss = tf.reduce_mean(binary_cross_entropy(tf.ones_like(s_f), s_f))
            #dis_loss = tf.math.log(s_r) + (tf.math.log(1 - s_w) + tf.math.log(1 - s_f))/2

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
    model.save_weights(path)


def load_model_weights(model, path):
    model.load_weights(path)

def visualize_generation_results(model, captions):
    """
    param model     : the trained generator model 
    param captions  : a list of captions
    """
    n = len(captions)
        
    fig = plt.figure(figsize=(1, n))

    gspec = gridspec.GridSpec(n, 1)
    gspec.update(wspace=0.05, hspace=0.5)


    for idx, cap in enumerate(captions):
        z = tf.random.normal([1, 128])
        print('vis caption shape', np.expand_dims(cap, axis=0).shape)
        generated_img = model(z, tf.expand_dims(cap, axis=0)).numpy()

        ax = plt.subplot(gspec[idx])
        
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        ax.set_title(cap)
        print("Max val", tf.math.reduce_max(generated_img))
        plt.imshow(tf.reshape(generated_img, (64, 64, 3)))

    plt.show()

    ## save to pdf (in case plt interactive window doesn't work)
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", "visualize_results.pdf")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ## get data
    #annFile_train='{}/annotations/instances_{}.json'.format('..','train2014')
    (train_images, train_captions) = get_data()
    print(train_images.shape)
    print(train_captions.shape)
    #annFile_test='{}/annotations/instances_{}.json'.format('..','test2014')
    #(test_images, test_captions) = get_data()

    generator = Generator()
    discriminator = Discriminator()
    
    train(generator, discriminator, train_images, train_captions)
    ## train model
    save_model_weights(generator, 'generator_weights.h5')
    save_model_weights(discriminator, 'discriminator_weights.h5')
    visualize_generation_results(generator, train_captions[0:100])

    ## test model


if __name__ == "__main__":

    ##args = parseArguments()
    main()