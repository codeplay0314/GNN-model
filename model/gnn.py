import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from utils import SelfAttention
import utils.data_loader as data_loader
from HGAT import HyperGraphAttentionLayer as HGAT

# Load Data
x_train, y_train, x_test, y_test, set_size, users, videos, edges = \
    data_loader.get(os.path.join(os.getcwd(), 'data/raw_15m.txt'))

print(set_size)

x_train = tf.convert_to_tensor(x_train.astype(np.int32))
y_train = tf.convert_to_tensor(y_train)
x_test = tf.convert_to_tensor(x_test.astype(np.int32))
y_test = tf.convert_to_tensor(y_test)

# Embedding Layer
EMBEDDING_DIM = 32
cat_embedding = layers.Embedding(set_size[2], EMBEDDING_DIM)
age_embedding = layers.Embedding(set_size[3], EMBEDDING_DIM)
len_embedding = layers.Embedding(set_size[4], EMBEDDING_DIM)
play_embedding = layers.Embedding(set_size[5], EMBEDDING_DIM)
src_embedding = layers.Embedding(set_size[6], EMBEDDING_DIM)
dep_embedding = layers.Embedding(set_size[7], EMBEDDING_DIM)
embedding_layers = [cat_embedding, age_embedding, len_embedding, play_embedding, src_embedding, dep_embedding]

# GNN
OUTPUT_DIM = EMBEDDING_DIM
HIDDEN_UNITS = 8
NUM_HEADS = 1
NUM_LAYERS = 3

gnn_layer = HGAT.HyperGraphAttentionLayer(
    embedding_layers,
    users,
    videos,
    edges,
    output_dim=OUTPUT_DIM,
    hidden_units=HIDDEN_UNITS,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS)

self_attention_layer = SelfAttention.MultiHeadSelfAttentionLayer(units=8, num_heads=4)

# Model
inputs = keras.Input(shape=(x_train.shape[1]), dtype="int32")

# user_emb, svid_emb, tvid_emb = gnn_layer(inputs)
user_emb = gnn_layer(inputs)

x = tf.stack([
    user_emb,
    # svid_emb,
    # tvid_emb,
    cat_embedding(inputs[:,3]),
    age_embedding(inputs[:,4]),
    len_embedding(inputs[:,5]),
    play_embedding(inputs[:,6]),
    cat_embedding(inputs[:,7]),
    age_embedding(inputs[:,8]),
    len_embedding(inputs[:,9]),
    play_embedding(inputs[:,10]),
    src_embedding(inputs[:,11]),
    dep_embedding(inputs[:,12])
    ], axis=1)
# x = self_attention_layer(x)
x = tf.unstack(x, axis=-1)
x = tf.concat(x, axis=-1)

x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(32, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid", name="predictions")(x)
model = tf.keras.Model(inputs, outputs)

model.summary()

# Train
BATCH_SIZE = 200
NUM_EPOCHS = 1000

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
plt.savefig("gnn_train.png")
plt.show()