import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Sequential
from keras.layers import Dropout
from keras.models import Model

import keras
from keras.datasets import mnist
from keras.datasets import cifar10

from tqdm import tqdm_notebook as tqdm
from keras.layers.advanced_activations import LeakyReLU

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

from numpy import expand_dims
from numpy import mean
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from keras import backend
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.initializers import RandomNormal
from keras.constraints import Constraint

# from keras.optimizers import Optimizer
# from tensorflow.keras import backend 
from keras.optimizers import Adam, SGD

from tqdm.notebook import tqdm
from functools import partial
from matplotlib.image import imread

# The Players class defines the two-player game and all necessary update functions
# x - min player
# y - max player
# f - a function of x, y to create new entity z
# u_x - function to take gradient update step of min player x
# u_y - function to take gradient update step of max player y
# c_x - function to change value of x to x_new
# c_y - function to change value of y to y_new
# p_x - function to perturb value of x along a random normal direction

class Players:
    def __init__(self, x, y, f, u_x, u_y, c_x, c_y, p_x):
        self.x = x
        self.y = y
        if f == None:
            self.z = None
        else:
            self.z = f(x, y)
        self.u_x = u_x
        self.u_y = u_y
        self.c_x = c_x
        self.c_y = c_y
        self.p_x = p_x
    
    def value(self, f):
        return f(self.x, self.y)

    def get_x(self):
        return self.x
                
    def get_y(self):
        return self.y       

    def update_x(self):
        self.x = self.u_x(self.x, self.y, self.z)
        return self.x
        
    def update_y(self):
        self.y = self.u_y(self.x, self.y, self.z)
        return self.y
    
    def change_x(self, x_new):
        self.x = self.c_x(self.x, x_new)
                
    def change_y(self, y_new):
        self.y = self.c_y(self.y, y_new)

    def perturb_x(self):
        self.x = self.p_x(self.x)
        return self.x
    

# img = imread('image_digit_1.png')
# img = img.reshape(400)

batch_size = 128

def getGDopt(lr = 0.01):
    return SGD(lr)

# Load Mnist data   
def load_data(filter=True):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5)/127.5
    
    x_train = x_train.reshape(len(x_train), 784)
    
    # restricting to digits 0 and 1
    if filter:
        train_filter = np.where((y_train == 0 ) | (y_train == 1))
        x_train, y_train = x_train[train_filter], y_train[train_filter]
        
    return (x_train, y_train, x_test, y_test)

# (X_train, y_train, X_test, y_test) = load_data()
# X_train = np.array([img]*128)

# Create generator network with preferred optimization function
def create_generator(OUTPUT_SIZE, opt=getGDopt(), INPUT_SIZE=100):
    generator=Sequential()
    generator.add(Dense(units=256,input_dim=INPUT_SIZE))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(units=512))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(units=1024))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(units=OUTPUT_SIZE, activation='tanh'))
    
    generator.compile(loss='binary_crossentropy', optimizer=opt)
    return generator

# Create discriminator network with preferred optimization function
def create_discriminator(INPUT_SIZE, opt=getGDopt()):
    discriminator=Sequential()
    discriminator.add(Dense(units=1024,input_dim=INPUT_SIZE))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
       
    discriminator.add(Dense(units=512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
       
    discriminator.add(Dense(units=256))
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Dense(units=1, activation='sigmoid'))
    
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=opt)
    return discriminator

def create_gan(generator, discriminator, opt=getGDopt(lr=0.01)):
    discriminator.trainable=False
    input_size = int(generator.input.shape[1])
    gan_input = Input(shape=(input_size,))
    x = generator(gan_input)
    gan_output= discriminator(x)
    gan= Model(inputs=gan_input, outputs=gan_output)
    
    gan.compile(loss='binary_crossentropy', optimizer=opt)

    return gan

# gradient update steps for discriminator
def take_discriminator_steps(generator, discriminator, gan, X_train, k=1, NOISE_SIZE=100):
    for _ in range(k):
        noise= np.random.normal(0,1, [batch_size, NOISE_SIZE])
        generated_images = generator.predict(noise)

        image_batch =X_train[np.random.randint(low=0,high=X_train.shape[0],size=batch_size)]

        X= np.concatenate([image_batch, generated_images])

        y_dis = np.zeros(2*batch_size)
        y_dis[:batch_size] = 1

        discriminator.trainable=True
        loss = discriminator.train_on_batch(X, y_dis)

    return discriminator

# perturbing weights of the generator
def perturb_generator(generator, sigma=0.001):
    weights, u = [], []
    for wt in generator.get_weights():
        u.append(np.random.normal(0, 1, wt.shape))
        wt = wt + u[-1] * sigma                
        weights.append(wt)
    
    generator.set_weights(weights)
        
    return generator

# gradient update steps for generator
def take_generator_steps(generator, discriminator, gan, NOISE_SIZE=100):
    noise= np.random.normal(0,1, [batch_size, NOISE_SIZE])
    generated_images = generator.predict(noise)

#     gan = create_gan(discriminator, generator)
    
    y_gen = np.ones(batch_size)
    discriminator.trainable=False
    gan.train_on_batch(noise, y_gen)
    
    return generator

def change_network(modela, modelb):
    modela.set_weights(modelb.get_weights())
    return modela
    
def wasserstein_loss(y_true, y_pred):
    return backend.mean(y_true * y_pred)

def getLoss(generator, discriminator, X_train, NOISE_SIZE=100):
    
    noise= np.random.normal(0,1, [batch_size, NOISE_SIZE])
    image_batch =X_train[np.random.randint(low=0,high=X_train.shape[0],size=batch_size)]
    generated_images = generator.predict(noise)
    
    probabilities_1 = discriminator.predict(image_batch).reshape(len(image_batch))
    probabilities_1 = np.log(probabilities_1)
    
    probabilities_2 = discriminator.predict(generated_images).reshape(len(generated_images))
    probabilities_2 = np.log(1 - probabilities_2)

    return np.mean(probabilities_1) + np.mean(probabilities_2)

def getLossFixedBatch(generator, discriminator, image_batch, generated_images):
    
    probabilities_1 = discriminator.predict(image_batch).reshape(len(image_batch))
    probabilities_1 = np.log(probabilities_1) / np.log(2)
    
    probabilities_2 = discriminator.predict(generated_images).reshape(len(generated_images))
    probabilities_2 = np.log(1 - probabilities_2) / np.log(2)

    return np.mean(probabilities_1) + np.mean(probabilities_2)

# Creating a GAN player object
def create_GAN_player():
    ganPlayer = Players(create_generator(), create_discriminator(), create_gan, take_generator_steps, take_discriminator_steps, change_network, change_network, perturb_generator)    
    return ganPlayer
    

def plot_generated_images(epoch, generator, folder="", save = False, image_shape=(28,28), examples=100, dim=(10,10), figsize=(10,10),name=""):
    noise= np.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples,image_shape[0], image_shape[1])
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    if save:
        plt.savefig(folder+name %epoch)

######################################################################


# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# select images
	X = dataset[ix]
	# generate class labels, -1 for 'real'
	y = -ones((n_samples, 1))
	return X, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = generator.predict(x_input)
	# create class labels with 1.0 for 'fake'
	y = ones((n_samples, 1))
	return X, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, latent_dim, n_samples=100):
	# prepare fake examples
	X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
	# scale from [-1,1] to [0,1]
	X = (X + 1) / 2.0
	# plot images
	for i in range(10 * 10):
		# define subplot
		plt.subplot(10, 10, 1 + i)
		# turn off axis
		plt.axis('off')
		# plot raw pixel data
		plt.imshow(X[i, :, :, 0], cmap='gray_r')
	# save plot to file
	filename1 = 'generated_plot_%04d.png' % (step+1)
	plt.savefig(filename1)
	plt.close()
	# save the generator model
	filename2 = 'model_%04d.h5' % (step+1)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))

# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, d2_hist, g_hist):
	# plot history
	plt.plot(d1_hist, label='crit_real')
	plt.plot(d2_hist, label='crit_fake')
	plt.plot(g_hist, label='gen')
	plt.legend()
	plt.savefig('plot_line_plot_loss.png')
	plt.close()
