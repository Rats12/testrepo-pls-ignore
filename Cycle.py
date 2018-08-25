
# coding: utf-8

# In[25]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.models import Model
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, Deconvolution2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.optimizers import Adam
from data_loader import DataLoader
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import numpy as np
from PIL import Image
import argparse
import math


# In[26]:

dataset_name = 'apple2orange'
data_loader = DataLoader(dataset_name=dataset_name,
                                      img_res=(128, 128))


        # Calculate output shape of D (PatchGAN)
#patch = int(128 / 2**4)
#disc_patch = (patch, patch, 1)
def discriminator():
    model = Sequential()
    model.add(
            Conv2D(64, (4, 4),
            padding='same',
            input_shape=(128, 128, 3))
            )
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (4, 4)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (4, 4)))
    model.add(Activation('tanh'))
    model.add(Conv2D(512, (4, 4)))
    model.add(Activation('tanh'))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    #model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
    


# In[27]:


#diss=discriminator()
#print(diss.summary())


# In[28]:


lambda_cycle = 10.0                    
lambda_id = 0.1 * lambda_cycle
optimizer = Adam(0.0002, 0.5)


# In[29]:


def residue(input1,input2,g):
    def s1():
        
        l0=Conv2D(g,4,strides=2,padding='same')(input2)
        l1=Activation('relu')(l0)
        l2=UpSampling2D(2)(l1)
        l3=Conv2D(int(g/2),4,strides=1,padding='same')(l2)
        
        return l3
    x=Concatenate()([input1,s1()])
    return x


# In[30]:


def generator():
    input_shape=(128, 128, 3)
    l0 = Input(shape=input_shape)
    l1=Conv2D(32,(2,2),strides=2,padding='same')(l0)
    l2=Activation('relu')(l1)
    l3=Conv2D(64,(2,2),strides=2,padding='same')(l2)
    l4=Activation('relu')(l3)    
    l5=Conv2D(128,(2,2),strides=2,padding='same')(l4)
    l6=Activation('relu')(l5)
    l7=Conv2D(256,(2,2),strides=2,padding='same')(l6)
    l8=Activation('relu')(l7)
    l9=UpSampling2D(2)(l8)
    g1=256
    l10=residue(l6,l9,int(g1))
    l11=UpSampling2D(2)(l10)
    g2=128
    l12=residue(l4,l11,int(g2))
    l13=UpSampling2D(2)(l12)
    g3=64
    l14=residue(l2,l13,int(g3))
    l15=UpSampling2D(2)(l14)
    l16=Conv2D(3,(2,2),strides=1,padding='same')(l15)
    model = Model(inputs=l0, outputs=l16)
    
    return model        
    
    
   
 


# In[31]:


#g=generator()
#print(g.summary())


# In[33]:



img_A=Input(shape=(128,128,3))
img_B=Input(shape=(128,128,3))

genA=generator()

genB=generator()
discA=discriminator()
discA.compile(loss='mse',
             optimizer=optimizer,
             metrics=['accuracy'])

discB=discriminator()

discB.compile(loss='mse',
             optimizer=optimizer,
             metrics=['accuracy'])


fake_B = genA(img_A)
fake_A = genB(img_B)

reconstr_A = genB(fake_B)
reconstr_B = genA(fake_A)

img_A_id = genB(img_A)
img_B_id = genA(img_B)

discA.trainable = False
discB.trainable = False

valid_A = discA(fake_A)
valid_B = discB(fake_B)

combined = Model(inputs=[img_A, img_B],
                              outputs=[ valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        img_A_id, img_B_id ])

combined.compile(loss=['mse', 'mse',
                            'mae', 'mae',
                            'mae', 'mae'],
                    loss_weights=[  1, 1,
                                    lambda_cycle, lambda_cycle,
                                    lambda_id, lambda_id ],
                        optimizer=optimizer)



# In[34]:


def train(epochs, batch_size=1, sample_interval=50):

    start_time = datetime.datetime.now()
    print("xe")
        # Adversarial loss ground truths
    valid = np.ones(batch_size) 
    fake = np.zeros(batch_size)

    for epoch in range(epochs):
        for batch_i, (imgs_A, imgs_B) in enumerate(data_loader.load_batch(batch_size)):

                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
            fake_B = genA.predict(imgs_A)
            fake_A = genB.predict(imgs_B)
            discA.trainable = True
            discB.trainable = True

                # Train the discriminators (original images = real / translated = Fake)
            discA_loss_real = discA.train_on_batch(imgs_A, valid)
            discA_loss_fake = discA.train_on_batch(fake_A, fake)
            discA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

            discB_loss_real = discB.train_on_batch(imgs_B, valid)
            discB_loss_fake = discB.train_on_batch(fake_B, fake)
            discB_loss = 0.5 * np.add(discB_loss_real, discB_loss_fake)

                # Total disciminator loss
            disc_loss = 0.5 * np.add(discA_loss, discB_loss)


                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
            gen_loss = combined.train_on_batch([imgs_A, imgs_B],
                                                        [valid, valid,
                                                        imgs_A, imgs_B,
                                                        imgs_A, imgs_B])

            elapsed_time = datetime.datetime.now() - start_time
            print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s "                                                                         % ( epoch, epochs,
                                                                            batch_i, data_loader.n_batches,
                                                                            d_loss[0], 100*d_loss[1],
                                                                            g_loss[0],
                                                                            np.mean(g_loss[1:3]),
                                                                            np.mean(g_loss[3:5]),
                                                                            np.mean(g_loss[5:6]),
                                                                            elapsed_time))

                # If at save interval => save generated image samples
            if batch_i % sample_interval == 0:
                esample_images(epoch, batch_i)


# In[35]:


def sample_images( epoch, batch_i):
    os.makedirs('images/%s' % dataset_name, exist_ok=True)
    r, c = 2, 3

    imgs_A = data_loader.load_data(domain="A", batch_size=1, is_testing=True)
    imgs_B = data_loader.load_data(domain="B", batch_size=1, is_testing=True)

       # Demo (for GIF)
       #imgs_A = self.data_loader.load_img('datasets/apple2orange/testA/n07740461_1541.jpg')
       #imgs_B = self.data_loader.load_img('datasets/apple2orange/testB/n07749192_4241.jpg')

       # Translate images to the other domain
    fake_B = genA.predict(imgs_A)
    fake_A = genB.predict(imgs_B)
       # Translate back to original domain
    reconstr_A = genB.predict(fake_B)
    reconstr_B = genA.predict(fake_A)

    gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])

       # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    titles = ['Original', 'Translated', 'Reconstructed']
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt])
            axs[i, j].set_title(titles[j])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("images/%s/%d_%d.png" % (dataset_name, epoch, batch_i))
    plt.close()

