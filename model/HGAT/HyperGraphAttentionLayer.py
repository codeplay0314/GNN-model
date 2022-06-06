from tensorflow import keras
from tensorflow.keras import layers
from . import MultiHeadHyperGraphAttention as MHHGAT
import tensorflow as tf

class HyperGraphAttentionLayer(layers.Layer):
    def __init__(
        self,
        embedding_layers,
        users,
        videos,
        edges,
        output_dim=64,
        hidden_units=32,
        num_heads=3,
        num_layers=3,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_layers = embedding_layers
        self.users = tf.convert_to_tensor(users, dtype=tf.int32)
        self.videos = tf.convert_to_tensor(videos, dtype=tf.int32)
        self.edges = tf.convert_to_tensor(edges, dtype=tf.int32)
        self.hyper_edges = tf.convert_to_tensor(edges.iloc[:, 0:3])

        self.state_dim = hidden_units * num_heads

        self.preprocess_user = layers.Dense(self.state_dim, activation="relu")
        self.preprocess_video = layers.Dense(self.state_dim, activation="relu")
        self.preprocess_edge = layers.Dense(self.state_dim, activation="relu")

        self.attention_layers = [
            MHHGAT.MultiHeadHyperGraphAttention(hidden_units, num_heads) for _ in range(num_layers)
        ]
        self.batch_norm = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)

        self.dropout1 = layers.Dropout(0.5)
        # self.dropout2 = layers.Dropout(0.5)
        self.user_out = layers.Dense(output_dim)
        # self.video_out = layers.Dense(output_dim)
        
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        self.user_embedding = self.add_weight(
            shape=(self.users.shape[0], self.state_dim),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="user_embedding",
        )
        self.built = True

    def call(self, inputs):
        user_states = self.user_embedding
        video_states = tf.concat([
            # self.embedding_layers[1](self.video_states[:,0]),
            self.embedding_layers[0](self.videos[:,1]),
            self.embedding_layers[2](self.videos[:,2])], axis=1)
        edge_states = tf.concat([
            tf.gather(self.user_embedding, self.edges[:,0]),
            # self.embedding_layers[1](self.edge_states[:,1]),
            # self.embedding_layers[1](self.edge_states[:,2]),
            self.embedding_layers[0](self.edges[:,3]),
            self.embedding_layers[1](self.edges[:,4]),
            self.embedding_layers[2](self.edges[:,5]),
            self.embedding_layers[3](self.edges[:,6]),
            self.embedding_layers[0](self.edges[:,7]),
            self.embedding_layers[1](self.edges[:,8]),
            self.embedding_layers[2](self.edges[:,9]),
            self.embedding_layers[3](self.edges[:,10]),
            self.embedding_layers[4](self.edges[:,11]),
            self.embedding_layers[5](self.edges[:,12])], axis=1)
        
        user_states = self.preprocess_user(user_states)
        video_states = self.preprocess_video(video_states)
        edge_states = self.preprocess_edge(edge_states)

        for attention_layer in self.attention_layers:
            ## // Only User Node
            # new_user_states, new_video_states = attention_layer([user_states, video_states, edge_states, self.hyper_edges])
            # user_states = user_states + self.dropout1(new_user_states)
            # video_states = video_states + self.dropout2(new_video_states)
            new_user_states = attention_layer([user_states, video_states, edge_states, self.hyper_edges])
            user_states = user_states + self.dropout1(new_user_states)
        
        user_states = self.batch_norm(user_states)
        
        self.user_embedding.assign(user_states)

        user_states = self.user_out(user_states)
        # video_states = self.video_out(video_states)

        user_indices = inputs[:, 0]
        # svid_indices = inputs[:, 1]
        # tvid_indices = inputs[:, 2]

        # return [tf.gather(user_states, user_indices), tf.gather(video_states, svid_indices), tf.gather(video_states, tvid_indices)]
        return tf.gather(user_states, user_indices)
