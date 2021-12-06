import tensorflow as tf
from models import Generator, Discriminator
from preprocessing import get_data
import random
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def train(gen_model, dis_model, imgs, captions):
    ## TODO: Write the training loop for 1 epoch of the model

    total_gen_loss = 0
    total_dis_loss = 0

    ## not batched?
    for img, caption in zip(imgs, captions):
            z = tf.random.normal([1, 128])
            fimg = gen_model(z, caption)
            rcap = captions[random.randint(0, len(captions))]
            s_r = dis_model(img, caption)
            s_w = dis_model(img, rcap)
            s_f = dis_model(fimg, caption)
            with tf.gradientTape() as tape:
                gen_loss = gen_model.loss(s_f)
                dis_loss = dis_model.loss(s_r, s_w, s_f)

                
            gen_gradients = tape.gradient(gen_loss, gen_model.trainable_variables)
            gen_model.optimizer.apply_gradients(zip(gen_gradients, gen_model.trainable_variables))
            dis_gradients = tape.gradient(dis_loss, dis_model.trainable_variables)
            dis_model.optimizer.apply_gradients(zip(dis_gradients, dis_model.trainable_variables))

            total_gen_loss += gen_loss
            total_dis_loss += dis_loss

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
            gen_loss = tf.math.log(s_f)
            dis_loss = tf.math.log(s_r) + (tf.math.log(1 - s_w) + tf.math.log(1 - s_f))/2
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
        generated_img = model(z, cap).numpy()

        ax = plt.subplot(gspec[idx])
        
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        ax.set_title(cap)

        plt.imshow(generated_img)

    plt.show()

    ## save to pdf (in case plt interactive window doesn't work)
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", "visualize_results.pdf")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ## get data
    annFile_train='{}/annotations/instances_{}.json'.format('..','train2014')
    (train_images, train_captions) = get_data(annFile_train)
    annFile_test='{}/annotations/instances_{}.json'.format('..','test2014')
    (test_images, test_captions) = get_data(annFile_test)

    generator = Generator()
    discriminator = Discriminator()
    
    train(generator, discriminator, train_images, train_captions)
    ## train model
    save_model_weights(generator, 'generator_weights.h5')
    save_model_weights(generator, 'discriminator_weights.h5')
    visualize_generation_results(generator, test_captions[0:10])

    ## test model


if __name__ == "__main__":

    ##args = parseArguments()
    main()