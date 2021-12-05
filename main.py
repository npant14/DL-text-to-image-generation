import tensorflow as tf
from models import Generator, Discriminator
from preprocessing import get_data

def train(gen_model, dis_model):
    ## TODO: Write the training loop for 1 epoch of the model

    total_gen_loss = 0
    total_dis_loss = 0
    imgs, captions = get_data("file.txt")
    for img, caption in zip(imgs, captions):
        with tf.gradientTape() as tape:
            fimg = gen_model(img, caption)
            out = dis_model(fimg, caption)
            gen_loss = gen_model.loss(fimg)
            dis_loss = dis_model.loss(out)
            total_gen_loss += gen_loss
            total_dis_loss += dis_loss
            gen_gradients = tape.gradient(gen_loss, gen_model.trainable_variables)
            gen_model.optimizer.apply_gradients(zip(gen_gradients, gen_model.trainable_variables))
            dis_gradients = tape.gradient(dis_loss, dis_model.trainable_variables)
            dis_model.optimizer.apply_gradients(zip(dis_gradients, dis_model.trainable_variables))
    return total_gen_loss, total_dis_loss


def test(model):
    ## TODO: Write the testing loop
    pass

def save_model_weights(model):
    ## TODO: Write to save the weights
    pass

def load_model_weights(model):
    ## TODO: Write to load the weights
    pass

def main():
    pass
    ## get data
    generator = Generator()
    discriminator = Discriminator()
    
    train(generator, discriminator)
    ## train model

    ## test model


if __name__ == "__main__":

    ##args = parseArguments()
    main()