
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import utils.data_loader as data_loader
from HGAT import HyperGraphAttentionLayer as HGAT
from utils import SelfAttention
import matplotlib.pyplot as plt

# Load Data
x_train, y_train, x_test, y_test, set_size, user_states, video_states, edges = \
    data_loader.get(os.path.join(os.getcwd(), 'data/raw_15m.txt'))

x_train = tf.convert_to_tensor(x_train.astype(np.int32))
y_train = tf.convert_to_tensor(y_train)
x_test = tf.convert_to_tensor(x_test.astype(np.int32))
y_test = tf.convert_to_tensor(y_test)

# Model

inputs = keras.Input(shape=(x_train.shape[1]), dtype="int32")
x = inputs[:, 3:]

outputs = layers.Dense(1, activation="sigmoid", name="predictions")(x)
model = tf.keras.Model(inputs, outputs)

model.summary()

# Train
LEARNING_RATE = 3e-1
MOMENTUM = 0.9

BATCH_SIZE = 128
NUM_EPOCHS = 1000
VALIDATION_SPLIT = 0.1

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=[keras.metrics.AUC(name='auc')]
)
history = model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    validation_data=(x_test, y_test),
    epochs=NUM_EPOCHS,
    callbacks=[EarlyStopping(monitor='val_auc', patience=20, verbose=2, mode='max', restore_best_weights=True)]
)
model.evaluate(x_test, y_test)

# Plot
keras.utils.plot_model(model, show_dtype=True, show_shapes=True)
plt.figure(figsize=(10, 6))
plt.plot(history.history["auc"], label="train AUC")
plt.plot(history.history["val_auc"], label="valid AUC")
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("AUC", fontsize=16)
plt.legend(fontsize=16)
plt.savefig("mlp_train.png")
plt.show()