from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import math

class SelfAttentionLayer(layers.Layer):
    def __init__(
        self,
        units,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        self.dropout = layers.Dropout(0.5)
    
    def build(self, input_shape):
        self.WQ = self.add_weight(
            shape=(input_shape[-1], self.units),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="WQ",
        )
        self.WK = self.add_weight(
            shape=(input_shape[-1], self.units),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="WK",
        )
        self.WV = self.add_weight(
            shape=(input_shape[-1], self.units),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="WV",
        )

    def call(self, inputs):
        Q = tf.matmul(inputs, self.WQ)
        K = tf.matmul(inputs, self.WK)
        V = tf.matmul(inputs, self.WV)

        attention_scores = tf.matmul(Q, K, transpose_b=True)
        attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(self.units)))
        attention_scores = tf.nn.softmax(attention_scores)
        attention_scores = self.dropout(attention_scores)

        outputs = tf.matmul(tf.transpose(V, perm=[0,2,1]), tf.transpose(attention_scores, perm=[0,2,1]))
        outputs = tf.transpose(outputs, perm=[0,2,1])

        return outputs

class MultiHeadSelfAttentionLayer(layers.Layer):
    def __init__(self, units, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.attention_layers = [SelfAttentionLayer(units) for _ in range(num_heads)]

    def call(self, inputs):
        # Obtain outputs from each attention head
        outputs = [
            attention_layer(inputs)
            for attention_layer in self.attention_layers
        ]
        outputs = tf.concat(outputs, axis=-1)

        return tf.nn.relu(outputs)
