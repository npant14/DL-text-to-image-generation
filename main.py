import tensorflow as tf
from models import Generator, Descriminator

def train(gen_model, des_model):
    ## TODO: Write the training loop for 1 epoch of the model

    ## currently all placeholder
    latent = tf.random.uniform([1, 512])
    text = tf.random.uniform([1,1024])
    fimg = gen_model(latent, text)

    out = des_model(fimg, text)
    print(out.shape)


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
    descriminator = Descriminator()
    
    train(generator, descriminator)
    ## train model

    ## test model


if __name__ == "__main__":

    ##args = parseArguments()
    main()