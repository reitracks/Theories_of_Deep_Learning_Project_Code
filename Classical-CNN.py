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

dataset = tf.keras.utils.image_dataset_from_directory(
  directory = x_dir,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  shuffle=True, 
  seed=123 
)

AUTOTUNE = tf.data.AUTOTUNE

def normalize_img(image, label):
  # Cast to float32 and rescale image values from [0, 255] to [0, 1]
  image = tf.cast(image, tf.float32) / 255.0
  return image, label

dataset = dataset.map(normalize_img)

train = dataset.take(200)
train = train.cache().prefetch(buffer_size=AUTOTUNE)
#train = ds.take(189)
val = dataset.skip(200)
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
      layers.Dense(100, activation='relu'),
      ])

    self.final = tf.keras.Sequential([
      layers.Dense(50, activation='relu', input_shape=(100,)),
      layers.Dense(5),
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.final(encoded)
    return decoded
  
checkpoint_path = "file-location"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model = Denoise()
model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])
model.summary()
history = model.fit(train,
            epochs=30,
            shuffle=True,
            validation_data=(val),
            callbacks=[cp_callback])

# Access the metrics from the history object
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(1, len(acc) + 1) # Epochs usually start from 1 for plotting

# Plotting the Training and Validation Accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)

# Plotting the Training and Validation Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.show()
