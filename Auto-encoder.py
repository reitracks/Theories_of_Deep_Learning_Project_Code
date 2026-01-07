import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

batch_size = 30
img_height = 80
img_width = 120

x_dir = 'folder-location'
y_dir = 'folder-location'

x_dataset = tf.keras.utils.image_dataset_from_directory(
  x_dir,
  label_mode=None, 
  image_size=(img_height, img_width),
  batch_size=batch_size,
  shuffle=True, 
  seed=123 
)

y_dataset = tf.keras.utils.image_dataset_from_directory(
  y_dir,
  label_mode=None, 
  image_size=(img_height, img_width),
  batch_size=batch_size,
  shuffle=True, 
  seed=123 
)

def normalize_img(image):
  # Cast to float32 and rescale from [0, 255] to [0, 1]
  return tf.cast(image, tf.float32) / 255.

x_dataset = x_dataset.map(normalize_img)
y_dataset = y_dataset.map(normalize_img)

ds = tf.data.Dataset.zip((x_dataset, y_dataset))

AUTOTUNE = tf.data.AUTOTUNE

train = ds.take(200)
train = train.cache().prefetch(buffer_size=AUTOTUNE)
#train = ds.take(189)
val = ds.skip(800)
val = val.cache().prefetch(buffer_size=AUTOTUNE)

class Denoise(Model):
  def __init__(self):
    super(Denoise, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(80, 120, 3)),
      layers.Conv2D(16, (5, 5), activation='relu', padding='same', strides=2),
      #layers.MaxPooling2D((2, 2), padding='same'),
      layers.Conv2D(8, (5, 5), activation='relu', padding='same', strides=2),
      #layers.MaxPooling2D((2, 2), padding='same'),
      layers.Flatten(),
      layers.Dense(500, activation='relu'),
      layers.Dense(300, activation='relu'),
      layers.Dense(25, activation='relu'),
      ])

    self.decoder = tf.keras.Sequential([
      layers.Dense(300, activation='relu', input_shape=(25,)),
      layers.Dense(500, activation='relu'),
      layers.Dense(20*30*8, activation='relu'),
      layers.Reshape((20, 30, 8)),
      layers.Conv2DTranspose(8, kernel_size=5, strides=2, activation='relu', padding='same'),
      #layers.UpSampling2D((2, 2)),
      layers.Conv2DTranspose(16, kernel_size=5, strides=2, activation='relu', padding='same'),
      #layers.UpSampling2D((2, 2)),
      layers.Conv2D(3, kernel_size=(5, 5), activation='sigmoid', padding='same')])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
  
checkpoint_path = "file-location"
#cp1 has latent dim 100
#cp2 has latent dim 50
#cp3 has latent dim 200
#cp4 has latent dim 25
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

def custom_mean_squared_error(y_true, y_pred):
    squared_difference = tf.square(tf.math.multiply(y_true - y_pred, y_true*10 + 0.1))
    return tf.reduce_mean(squared_difference, axis=-1)

autoencoder = Denoise()
autoencoder.compile(optimizer='adam', loss=custom_mean_squared_error)
autoencoder.encoder.summary()
history = autoencoder.fit(train,
                epochs=100,
                shuffle=True,
                validation_data=(val),
                callbacks=[cp_callback])

