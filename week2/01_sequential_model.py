# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

tf.config.experimental_run_functions_eagerly(True)

# %%
# Build convolutional neural network

model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((3, 3)),
    Flatten(),
    Dense(10, activation='softmax')
])
model.summary()

# %%
# Compile the model

opt = tf.keras.optimizers.Adam(learning_rate=0.005)
loss = tf.keras.losses.SparseCategoricalCrossentropy()
acc = tf.keras.metrics.SparseCategoricalAccuracy()
mae = tf.keras.metrics.MeanAbsoluteError()

model.compile(optimizer=opt,
              loss=loss,
              metrics=[acc, mae])


# %%
# Load the dataset

fashion_mnist_data = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist_data.load_data()

train_images.shape

labels = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]

# %%
# Show an image from the dataset

train_images = train_images / 255
test_images = test_images / 255

i = 0
img = train_images[i, :, :]
plt.imshow(img)
print(f"label: {labels[train_labels[i]]}")

# %%
# Fit the model

history = model.fit(train_images[..., np.newaxis], train_labels, epochs=8, batch_size=256)

# %%
# Plot training history

df = pd.DataFrame(history.history)
df.head()

loss_plot = df.plot(y='loss')
loss_plot.set(xlabel='Epochs', ylabel='Loss')

# %%
# Evaluate how the model performs with the test dataset

test_loss, test_accuracy, test_mae = model.evaluate(test_images, test_labels)

# %%
# Make predictions from the model

random_inx = np.random.choice(test_images.shape[0])
test_image = test_images[random_inx]
plt.imshow(test_image)
print(f"Actual: {labels[test_labels[random_inx]]}")

# %%
# View the predictions

predictions = model.predict(test_image[np.newaxis, ..., np.newaxis])
print(f"Prediction: {labels[np.argmax(predictions)]}")

# %%
